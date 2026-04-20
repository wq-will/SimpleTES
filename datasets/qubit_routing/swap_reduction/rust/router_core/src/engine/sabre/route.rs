use std::collections::{HashMap, HashSet, VecDeque};

use hashbrown::HashMap as HbHashMap;
use qiskit_circuit::dag_circuit::{DAGCircuit, DAGCircuitBuilder, NodeType, Wire};
use qiskit_circuit::operations::{ControlFlow, Operation, StandardGate};
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::{Block, BlocksMode, Qubit};
use qiskit_qasm3::{dumps_from_dag, loads_to_dag};
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::token_swapper::token_swapper;

use crate::candidate::{
    CircuitDagView, DagNodeKind, FrontLayerView, InitialLayoutContext, LayoutView, Policy,
    PolicyDagData, PolicyDagNodeData, RemainingDagView, RemainingNodeState, RngState,
    SwapSelectionContext, TopologyView,
};
use crate::engine::routing_target::RoutingTarget;
use crate::engine::sabre::dag::{InteractionKind, SabreDAG};
use crate::engine::sabre::layer::FrontLayer;
use crate::engine::sabre::layout::Layout;
use crate::engine::{RouteOptions, RouteOutput, RouterError};

const PRECOMPUTED_EXTENDED_SET_LOGICAL_CAP: usize = 256;

#[derive(Debug, Clone, Copy)]
enum RoutedItemKind {
    Simple,
    ControlFlow(usize),
}

#[derive(Debug, Clone)]
struct RoutedItem {
    initial_swaps: Option<Vec<(usize, usize)>>,
    node_id: usize,
    kind: RoutedItemKind,
}

#[derive(Debug, Clone)]
struct RoutingResult<'a> {
    policy_dag: PolicyDagData,
    dag: &'a DAGCircuit,
    initial_layout: Layout,
    final_layout: Layout,
    order: Vec<RoutedItem>,
    final_swaps: Vec<(usize, usize)>,
    control_flow: Vec<RoutingResult<'a>>,
    swap_count: usize,
}

impl RoutingResult<'_> {
    fn rebuild(&self, num_physical_qubits: usize) -> Result<DAGCircuit, RouterError> {
        let out = self
            .dag
            .physical_empty_like_with_capacity(
                num_physical_qubits,
                self.dag.num_ops().saturating_add(self.swap_count),
                self.dag
                    .dag()
                    .edge_count()
                    .saturating_add(self.swap_count.saturating_mul(2)),
                BlocksMode::Drop,
            )
            .map_err(|e| RouterError::Routing(format!("failed to allocate output DAG: {e}")))?;
        self.rebuild_onto(out)
    }

    fn rebuild_onto(&self, dag_out: DAGCircuit) -> Result<DAGCircuit, RouterError> {
        let apply_swap =
            |swap: (usize, usize), layout: &mut Layout, dag: &mut DAGCircuitBuilder| {
                layout.swap_physical(swap.0, swap.1);
                let qargs = [Qubit(swap.1 as u32), Qubit(swap.0 as u32)];
                let inst = PackedInstruction::from_standard_gate(
                    StandardGate::Swap,
                    None,
                    dag.insert_qargs(&qargs),
                );
                dag.push_back(inst).map(|_| ()).map_err(|e| {
                    RouterError::Routing(format!("failed to append swap during rebuild: {e}"))
                })
            };

        let mut apply_scratch = Vec::with_capacity(8);
        let mut apply_op =
            |inst: &PackedInstruction, layout: &Layout, dag: &mut DAGCircuitBuilder| {
                apply_scratch.clear();
                for q in self.dag.get_qargs(inst.qubits) {
                    apply_scratch.push(Qubit(layout.physical_of_logical(q.index()) as u32));
                }
                let new_inst = PackedInstruction {
                    qubits: dag.insert_qargs(&apply_scratch),
                    ..inst.clone()
                };
                dag.push_back(new_inst).map(|_| ()).map_err(|e| {
                    RouterError::Routing(format!("failed to append routed operation: {e}"))
                })
            };

        let mut dag = dag_out.into_builder();
        let mut layout = self.initial_layout.clone();
        let mut next_block = 0usize;

        for &dag_node in &self.policy_dag.initial_ops {
            let NodeType::Operation(inst) = &self.dag[dag_node] else {
                continue;
            };
            apply_op(inst, &layout, &mut dag)?;
        }

        for item in &self.order {
            if let Some(swaps) = &item.initial_swaps {
                for &swap in swaps {
                    apply_swap(swap, &mut layout, &mut dag)?;
                }
            }

            let Some((&dag_node, rest)) = self.policy_dag.nodes[item.node_id]
                .dag_indices
                .split_first()
            else {
                continue;
            };
            let NodeType::Operation(inst) = &self.dag[dag_node] else {
                return Err(RouterError::Routing(
                    "policy DAG node does not point to an operation".to_string(),
                ));
            };

            match item.kind {
                RoutedItemKind::Simple => {
                    apply_op(inst, &layout, &mut dag)?;
                }
                RoutedItemKind::ControlFlow(num_blocks) => {
                    let start = next_block;
                    let end = start.saturating_add(num_blocks);
                    if end > self.control_flow.len() {
                        return Err(RouterError::Routing(
                            "control-flow rebuild bookkeeping mismatch".to_string(),
                        ));
                    }
                    let mut rebuilt_blocks = Vec::with_capacity(num_blocks);
                    for block in &self.control_flow[start..end] {
                        rebuilt_blocks.push(block.rebuild(layout.num_physical())?);
                    }
                    next_block = end;

                    let explicit = self
                        .dag
                        .get_qargs(inst.qubits)
                        .iter()
                        .map(|q| q.index())
                        .collect::<HashSet<_>>();
                    let mut qargs = Vec::new();
                    let mut idle = Vec::new();
                    for phys in 0..layout.num_physical() {
                        let explicit_wire = layout
                            .logical_of_physical(phys)
                            .is_some_and(|logical| explicit.contains(&logical));
                        let active_in_block = rebuilt_blocks
                            .iter()
                            .any(|block| !block.is_wire_idle(Wire::Qubit(Qubit(phys as u32))));
                        if explicit_wire || active_in_block {
                            qargs.push(Qubit(phys as u32));
                        } else {
                            idle.push(Qubit(phys as u32));
                        }
                    }
                    for block in &mut rebuilt_blocks {
                        block.remove_qubits(idle.iter().copied()).map_err(|e| {
                            RouterError::Routing(format!(
                                "failed removing idle qubits from rebuilt control-flow block: {e}"
                            ))
                        })?;
                    }

                    let mut new_op = inst.op.control_flow().clone();
                    if !matches!(
                        &new_op.control_flow,
                        ControlFlow::BreakLoop | ControlFlow::ContinueLoop
                    ) {
                        if let Some(first) = rebuilt_blocks.first() {
                            new_op.num_qubits = first.num_qubits() as u32;
                        }
                    }

                    let blocks: Vec<Block> = rebuilt_blocks
                        .into_iter()
                        .map(|block| dag.add_block(block))
                        .collect();
                    let new_inst = PackedInstruction::from_control_flow(
                        new_op,
                        blocks,
                        dag.insert_qargs(&qargs),
                        inst.clbits,
                        inst.label.as_deref().cloned(),
                    );
                    dag.push_back(new_inst).map_err(|e| {
                        RouterError::Routing(format!(
                            "failed to append control-flow operation during rebuild: {e}"
                        ))
                    })?;
                }
            }

            for &extra_dag_node in rest {
                let NodeType::Operation(extra_inst) = &self.dag[extra_dag_node] else {
                    continue;
                };
                apply_op(extra_inst, &layout, &mut dag)?;
            }
        }

        for &swap in &self.final_swaps {
            apply_swap(swap, &mut layout, &mut dag)?;
        }

        if next_block != self.control_flow.len() {
            return Err(RouterError::Routing(
                "control-flow block sequence was not fully consumed during rebuild".to_string(),
            ));
        }

        Ok(dag.build())
    }
}

struct RouteState<'dag, 'policy, 'ctx, P: Policy> {
    policy_dag: &'policy PolicyDagData,
    dag: &'dag DAGCircuit,
    target: &'ctx RoutingTarget,
    policy: &'ctx mut P,
    layout: Layout,
    front_layer: FrontLayer,
    remaining_predecessors: Vec<usize>,
    node_state: Vec<RemainingNodeState>,
    rng: RngState,
    seed: u64,
    policy_swaps_since_progress: usize,
    last_policy_swap: Option<(usize, usize)>,
    max_swaps_without_progress: usize,
    order: Vec<RoutedItem>,
    control_flow: Vec<RoutingResult<'dag>>,
    swap_count: usize,
    ready_node_ids: Vec<usize>,
    ready_node_pos: Vec<usize>,
    pending_node_ids: Vec<usize>,
    pending_node_pos: Vec<usize>,
    remaining_two_qubit_node_ids: Vec<usize>,
    remaining_two_qubit_node_pos: Vec<usize>,
    front_layer_node_ids: Vec<usize>,
    front_layer_pairs: Vec<[usize; 2]>,
    precomputed_extended_set_logical_pairs: Vec<[usize; 2]>,
    extended_set_required_predecessors: Vec<usize>,
    extended_set_to_visit: Vec<usize>,
}

pub fn route_qasm3<P: Policy>(
    qasm3: &str,
    target: &RoutingTarget,
    policy: &mut P,
    options: RouteOptions,
) -> Result<RouteOutput, RouterError> {
    let dag = load_dag_via_python_qiskit(qasm3)?;
    if dag.num_qubits() > target.num_qubits {
        return Err(RouterError::Routing(format!(
            "circuit needs {} logical qubits but topology has {} physical qubits",
            dag.num_qubits(),
            target.num_qubits
        )));
    }

    let sabre = SabreDAG::from_dag(&dag)?;
    let policy_dag = build_policy_dag(&dag, &sabre)?;
    let topology = TopologyView::new(target);
    let circuit = CircuitDagView::new(&policy_dag);
    let mut rng = RngState::new(options.seed);
    let initial_layout = policy
        .choose_best_initial_layout(&InitialLayoutContext::new(topology, circuit), &mut rng)?;
    validate_initial_layout(&initial_layout, dag.num_qubits(), target)?;

    let layout_trials = build_layout_trials(
        &initial_layout,
        target,
        options.layout_trials.max(1),
        options.seed,
    );
    let layout_trial_seeds = trial_seeds(options.seed, layout_trials.len().max(1));

    let template = policy.clone();
    let mut best: Option<RoutingResult<'_>> = None;
    let mut first_error: Option<RouterError> = None;
    for (layout_index, mapping) in layout_trials.iter().enumerate() {
        for seed in trial_seeds(
            layout_trial_seeds[layout_index],
            options.routing_trials.max(1),
        ) {
            let mut trial_policy = template.clone();
            match route_with_initial_layout(
                &dag,
                &policy_dag,
                target,
                &mut trial_policy,
                Layout::from_logical_to_physical(mapping.clone(), target.num_qubits),
                seed,
                options.max_swaps_without_progress,
            ) {
                Ok(result) => {
                    if best
                        .as_ref()
                        .is_none_or(|cur| result.swap_count < cur.swap_count)
                    {
                        best = Some(result);
                    }
                }
                Err(err) => {
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }
    }
    let Some(result) = best else {
        return Err(first_error.unwrap_or_else(|| {
            RouterError::Routing("failed to produce any routing trial".to_string())
        }));
    };

    let routed_dag = result.rebuild(target.num_qubits)?;
    let depth = compute_depth_from_dag(&routed_dag);
    let twoq_count = count_twoq_ops(&routed_dag);
    let output_circuit = dump_dag_via_python_qiskit(&routed_dag)?;

    Ok(RouteOutput {
        swap_count: result.swap_count,
        depth,
        twoq_count,
        initial_mapping: result.initial_layout.logical_to_physical_map().to_vec(),
        output_circuit,
    })
}

fn route_with_initial_layout<'dag, 'policy, P: Policy>(
    dag: &'dag DAGCircuit,
    policy_dag: &'policy PolicyDagData,
    target: &RoutingTarget,
    policy: &mut P,
    layout: Layout,
    seed: u64,
    max_swaps_without_progress: usize,
) -> Result<RoutingResult<'dag>, RouterError> {
    let initial_layout = layout.clone();
    let node_count = policy_dag.nodes.len();
    let remaining_predecessors = policy_dag
        .nodes
        .iter()
        .map(|node| node.predecessors.len())
        .collect::<Vec<_>>();
    let pending_node_ids = (0..node_count).collect::<Vec<_>>();
    let pending_node_pos = (0..node_count).collect::<Vec<_>>();
    let mut remaining_two_qubit_node_ids = Vec::new();
    for (node_id, node) in policy_dag.nodes.iter().enumerate() {
        if node.kind == DagNodeKind::TwoQ {
            remaining_two_qubit_node_ids.push(node_id);
        }
    }
    let mut remaining_two_qubit_node_pos = vec![usize::MAX; node_count];
    for (pos, node_id) in remaining_two_qubit_node_ids.iter().copied().enumerate() {
        remaining_two_qubit_node_pos[node_id] = pos;
    }
    let mut state = RouteState {
        policy_dag,
        dag,
        target,
        policy,
        layout,
        front_layer: FrontLayer::new(target.num_qubits),
        remaining_predecessors,
        node_state: vec![RemainingNodeState::Pending; node_count],
        rng: RngState::new(seed),
        seed,
        policy_swaps_since_progress: 0,
        last_policy_swap: None,
        max_swaps_without_progress,
        order: Vec::new(),
        control_flow: Vec::new(),
        swap_count: 0,
        ready_node_ids: Vec::new(),
        ready_node_pos: vec![usize::MAX; node_count],
        pending_node_ids,
        pending_node_pos,
        remaining_two_qubit_node_ids,
        remaining_two_qubit_node_pos,
        front_layer_node_ids: Vec::new(),
        front_layer_pairs: Vec::new(),
        precomputed_extended_set_logical_pairs: Vec::new(),
        extended_set_required_predecessors: vec![0; node_count],
        extended_set_to_visit: Vec::new(),
    };

    state.update_route(policy_dag.first_layer.as_slice(), None)?;
    state.refresh_precomputed_extended_set();

    let mut routable_nodes = Vec::<usize>::new();
    while !state.front_layer.is_empty() {
        let mut current_swaps = Vec::<(usize, usize)>::new();
        while routable_nodes.is_empty()
            && current_swaps.len() <= state.max_swaps_without_progress.max(1)
        {
            let Some(best_swap) = state.choose_best_swap() else {
                break;
            };
            if !state.target.is_adjacent(best_swap.0, best_swap.1) {
                return Err(RouterError::Routing(format!(
                    "policy proposed invalid swap ({}, {}) that is not a coupling edge",
                    best_swap.0, best_swap.1
                )));
            }
            state.apply_swap(best_swap);
            current_swaps.push(best_swap);
            state.policy_swaps_since_progress = state.policy_swaps_since_progress.saturating_add(1);
            state.last_policy_swap = Some(best_swap);

            if let Some(node) = state.routable_node_on_qubit(best_swap.1) {
                routable_nodes.push(node);
            }
            if let Some(node) = state.routable_node_on_qubit(best_swap.0) {
                routable_nodes.push(node);
            }
        }

        if routable_nodes.is_empty() {
            for swap in current_swaps.drain(..).rev() {
                state.apply_swap(swap);
            }
            routable_nodes = state.force_enable_closest_node(&mut current_swaps)?;
        }

        routable_nodes.sort_unstable();
        routable_nodes.dedup();
        for node in &routable_nodes {
            state.front_layer.remove(node);
        }
        state.update_route(&routable_nodes, Some(current_swaps))?;
        state.refresh_precomputed_extended_set();
        state.policy_swaps_since_progress = 0;
        state.last_policy_swap = None;
        routable_nodes.clear();
    }

    Ok(RoutingResult {
        policy_dag: policy_dag.clone(),
        dag,
        initial_layout,
        final_layout: state.layout,
        order: state.order,
        final_swaps: Vec::new(),
        control_flow: state.control_flow,
        swap_count: state.swap_count,
    })
}

impl<P: Policy> RouteState<'_, '_, '_, P> {
    fn update_route(
        &mut self,
        nodes: &[usize],
        mut initial_swaps: Option<Vec<(usize, usize)>>,
    ) -> Result<(), RouterError> {
        let mut to_visit = nodes.iter().copied().collect::<VecDeque<_>>();
        while let Some(node_id) = to_visit.pop_front() {
            if self.node_state[node_id] == RemainingNodeState::Executed {
                continue;
            }

            let node_kind = self.policy_dag.nodes[node_id].kind;
            match node_kind {
                DagNodeKind::TwoQ => {
                    let qargs = self.policy_dag.nodes[node_id].qargs.clone();
                    let [a, b]: [usize; 2] = qargs.as_slice().try_into().map_err(|_| {
                        RouterError::Routing(format!(
                            "policy DAG node {node_id} marked TwoQ without exactly two qargs"
                        ))
                    })?;
                    let pa = self.layout.physical_of_logical(a);
                    let pb = self.layout.physical_of_logical(b);
                    if !self.target.is_adjacent(pa, pb) {
                        self.front_layer.insert(node_id, [pa, pb]);
                        self.node_state[node_id] = RemainingNodeState::Ready;
                        self.add_ready_node(node_id);
                        continue;
                    }
                }
                DagNodeKind::ControlFlow => {
                    let block_count = self.policy_dag.nodes[node_id].block_count;
                    self.route_control_flow_node(node_id)?;
                    self.record_execution(
                        node_id,
                        initial_swaps.take(),
                        RoutedItemKind::ControlFlow(block_count),
                        &mut to_visit,
                    );
                    continue;
                }
                DagNodeKind::Directive | DagNodeKind::SingleQ | DagNodeKind::MultiQ => {}
            }

            self.record_execution(
                node_id,
                initial_swaps.take(),
                RoutedItemKind::Simple,
                &mut to_visit,
            );
        }

        debug_assert!(
            initial_swaps.is_none(),
            "initial swaps must be consumed by at least one routed node"
        );
        Ok(())
    }

    fn route_control_flow_node(&mut self, node_id: usize) -> Result<(), RouterError> {
        let dag_node = self.policy_dag.nodes[node_id].dag_index;
        let NodeType::Operation(inst) = &self.dag[dag_node] else {
            return Err(RouterError::Routing(
                "control-flow policy node does not point to an operation".to_string(),
            ));
        };
        let block_layout =
            control_flow_block_layout(&self.layout, self.dag, inst, self.target.num_qubits)?;
        let Some(control_flow) = self.dag.try_view_control_flow(inst) else {
            return Err(RouterError::Routing(
                "control-flow policy node lost its control-flow view".to_string(),
            ));
        };

        for block_dag in control_flow.blocks() {
            let block_seed = self.seed;
            let mut block_policy = self.policy.clone();
            let block_sabre = SabreDAG::from_dag(block_dag)?;
            let block_policy_dag = build_policy_dag(block_dag, &block_sabre)?;
            let mut block_result = route_with_initial_layout(
                block_dag,
                &block_policy_dag,
                self.target,
                &mut block_policy,
                block_layout.clone(),
                block_seed,
                self.max_swaps_without_progress,
            )?;
            let restore = restore_layout_swaps(
                &block_result.final_layout,
                &block_layout,
                self.target,
                block_seed,
            )?;
            block_result.swap_count += restore.len();
            block_result.final_swaps = restore;
            block_result.final_layout = block_layout.clone();
            self.swap_count += block_result.swap_count;
            self.control_flow.push(block_result);
        }

        Ok(())
    }

    fn record_execution(
        &mut self,
        node_id: usize,
        initial_swaps: Option<Vec<(usize, usize)>>,
        kind: RoutedItemKind,
        to_visit: &mut VecDeque<usize>,
    ) {
        if let Some(swaps) = &initial_swaps {
            self.swap_count += swaps.len();
        }
        self.node_state[node_id] = RemainingNodeState::Executed;
        self.remove_pending_node(node_id);
        self.remove_ready_node(node_id);
        self.remove_remaining_two_qubit_node(node_id);
        self.order.push(RoutedItem {
            initial_swaps,
            node_id,
            kind,
        });

        for &succ in &self.policy_dag.nodes[node_id].successors {
            self.remaining_predecessors[succ] = self.remaining_predecessors[succ].saturating_sub(1);
            if self.remaining_predecessors[succ] == 0 {
                to_visit.push_back(succ);
            }
        }
    }

    fn apply_swap(&mut self, swap: (usize, usize)) {
        self.front_layer.apply_swap(swap);
        self.layout.swap_physical(swap.0, swap.1);
    }

    fn routable_node_on_qubit(&self, qubit: usize) -> Option<usize> {
        self.front_layer.qubits()[qubit]
            .and_then(|(node, other)| self.target.is_adjacent(qubit, other).then_some(node))
    }

    fn choose_best_swap(&mut self) -> Option<(usize, usize)> {
        self.refresh_front_layer_scratch();
        let topology = TopologyView::new(self.target);
        let layout = LayoutView::new(&self.layout);
        let circuit = CircuitDagView::new(self.policy_dag);
        let remaining = RemainingDagView::new(
            circuit,
            &self.node_state,
            &self.remaining_predecessors,
            &self.ready_node_ids,
            &self.pending_node_ids,
            &self.remaining_two_qubit_node_ids,
            &self.front_layer_node_ids,
        );
        let front_layer = FrontLayerView::new(
            circuit,
            layout,
            &self.front_layer_node_ids,
            &self.front_layer_pairs,
        );
        let ctx = if self.precomputed_extended_set_logical_pairs.is_empty() {
            SwapSelectionContext::new(
                topology,
                layout,
                circuit,
                remaining,
                front_layer,
                self.policy_swaps_since_progress,
                self.last_policy_swap,
            )
        } else {
            SwapSelectionContext::new_with_precomputed_extended_set(
                topology,
                layout,
                circuit,
                remaining,
                front_layer,
                &self.precomputed_extended_set_logical_pairs,
                self.policy_swaps_since_progress,
                self.last_policy_swap,
            )
        };
        self.policy.choose_best_swap(&ctx, &mut self.rng)
    }

    fn refresh_front_layer_scratch(&mut self) {
        self.front_layer_node_ids.clear();
        self.front_layer_pairs.clear();
        for (&node_id, qubits) in self.front_layer.iter() {
            self.front_layer_node_ids.push(node_id);
            self.front_layer_pairs.push(*qubits);
        }
    }

    fn add_ready_node(&mut self, node_id: usize) {
        tracked_insert(&mut self.ready_node_ids, &mut self.ready_node_pos, node_id);
    }

    fn remove_ready_node(&mut self, node_id: usize) {
        tracked_remove(&mut self.ready_node_ids, &mut self.ready_node_pos, node_id);
    }

    fn remove_pending_node(&mut self, node_id: usize) {
        tracked_remove(&mut self.pending_node_ids, &mut self.pending_node_pos, node_id);
    }

    fn remove_remaining_two_qubit_node(&mut self, node_id: usize) {
        tracked_remove(
            &mut self.remaining_two_qubit_node_ids,
            &mut self.remaining_two_qubit_node_pos,
            node_id,
        );
    }

    fn refresh_precomputed_extended_set(&mut self) {
        self.precomputed_extended_set_logical_pairs.clear();
        if PRECOMPUTED_EXTENDED_SET_LOGICAL_CAP == 0 {
            return;
        }
        if self.extended_set_required_predecessors.len() != self.remaining_predecessors.len() {
            self.extended_set_required_predecessors = vec![0; self.remaining_predecessors.len()];
        }
        self.extended_set_required_predecessors
            .copy_from_slice(&self.remaining_predecessors);
        self.extended_set_to_visit.clear();
        for (&node_id, _) in self.front_layer.iter() {
            self.extended_set_to_visit.push(node_id);
        }

        let mut i = 0usize;
        while i < self.extended_set_to_visit.len()
            && self.precomputed_extended_set_logical_pairs.len()
                < PRECOMPUTED_EXTENDED_SET_LOGICAL_CAP
        {
            let node_id = self.extended_set_to_visit[i];
            for &successor in &self.policy_dag.nodes[node_id].successors {
                debug_assert!(
                    self.extended_set_required_predecessors[successor] > 0,
                    "extended-set predecessor counter underflow on node {successor}"
                );
                self.extended_set_required_predecessors[successor] -= 1;
                if self.extended_set_required_predecessors[successor] == 0 {
                    let successor_node = &self.policy_dag.nodes[successor];
                    if successor_node.kind == DagNodeKind::TwoQ && successor_node.qargs.len() == 2 {
                        self.precomputed_extended_set_logical_pairs
                            .push([successor_node.qargs[0], successor_node.qargs[1]]);
                    }
                    self.extended_set_to_visit.push(successor);
                }
            }
            i += 1;
        }
    }

    fn force_enable_closest_node(
        &mut self,
        current_swaps: &mut Vec<(usize, usize)>,
    ) -> Result<Vec<usize>, RouterError> {
        let Some((&closest_node, qubits)) = self
            .front_layer
            .iter()
            .min_by_key(|(_, qubits)| self.target.distance(qubits[0], qubits[1]))
        else {
            return Err(RouterError::Routing(
                "front layer unexpectedly empty while forcing progress".to_string(),
            ));
        };

        let shortest_path = self.target.shortest_path(qubits[0], qubits[1]);
        if shortest_path.len() < 2 {
            return Err(RouterError::Routing(format!(
                "target has no shortest path between {} and {}",
                qubits[0], qubits[1]
            )));
        }

        let split = shortest_path.len() / 2;
        current_swaps.reserve(shortest_path.len().saturating_sub(2));
        for i in 0..split {
            current_swaps.push((shortest_path[i], shortest_path[i + 1]));
        }
        for i in 0..split.saturating_sub(1) {
            let end = shortest_path.len() - 1 - i;
            current_swaps.push((shortest_path[end], shortest_path[end - 1]));
        }
        for &swap in current_swaps.iter() {
            self.apply_swap(swap);
        }

        if current_swaps.len() > 1 {
            Ok(vec![closest_node])
        } else {
            let mut out = vec![closest_node];
            if let Some(swap) = current_swaps.first().copied() {
                for phys in [swap.0, swap.1] {
                    if let Some((node, _)) = self.front_layer.qubits()[phys] {
                        if node != closest_node && self.routable_node_on_qubit(phys).is_some() {
                            out.push(node);
                        }
                    }
                }
            }
            out.sort_unstable();
            out.dedup();
            Ok(out)
        }
    }
}

fn tracked_insert(store: &mut Vec<usize>, pos: &mut [usize], value: usize) {
    if pos[value] != usize::MAX {
        return;
    }
    pos[value] = store.len();
    store.push(value);
}

fn tracked_remove(store: &mut Vec<usize>, pos: &mut [usize], value: usize) {
    let idx = pos[value];
    if idx == usize::MAX {
        return;
    }
    let last = store
        .pop()
        .expect("tracked set storage must be non-empty when removing");
    if idx < store.len() {
        store[idx] = last;
        pos[last] = idx;
    }
    pos[value] = usize::MAX;
}

fn validate_initial_layout(
    mapping: &[usize],
    num_logical_qubits: usize,
    target: &RoutingTarget,
) -> Result<(), RouterError> {
    if mapping.len() != num_logical_qubits {
        return Err(RouterError::Routing(format!(
            "initial layout must map {num_logical_qubits} logical qubits, got {}",
            mapping.len()
        )));
    }

    let mut seen = vec![false; target.num_qubits];
    for (logical, &physical) in mapping.iter().enumerate() {
        if physical >= target.num_qubits {
            return Err(RouterError::Routing(format!(
                "initial layout maps logical qubit {logical} to out-of-range physical qubit {physical}"
            )));
        }
        if std::mem::replace(&mut seen[physical], true) {
            return Err(RouterError::Routing(format!(
                "initial layout maps multiple logical qubits to physical qubit {physical}"
            )));
        }
    }

    Ok(())
}

fn build_policy_dag(dag: &DAGCircuit, sabre: &SabreDAG) -> Result<PolicyDagData, RouterError> {
    let mut nodes = Vec::<PolicyDagNodeData>::with_capacity(sabre.nodes.len());
    for sabre_node in &sabre.nodes {
        let Some(&dag_index) = sabre_node.indices.first() else {
            return Err(RouterError::Routing(
                "sabre node unexpectedly missing DAG indices".to_string(),
            ));
        };
        let NodeType::Operation(inst) = &dag[dag_index] else {
            return Err(RouterError::Routing(
                "sabre node does not point to an operation".to_string(),
            ));
        };

        let qargs = dag
            .get_qargs(inst.qubits)
            .iter()
            .map(|q| q.index())
            .collect::<Vec<_>>();
        let cargs = dag
            .get_cargs(inst.clbits)
            .iter()
            .map(|c| c.index())
            .collect::<Vec<_>>();
        let block_count = dag
            .try_view_control_flow(inst)
            .map_or(0, |cf| cf.blocks().len());
        let kind = match &sabre_node.kind {
            InteractionKind::TwoQ(_) => DagNodeKind::TwoQ,
            InteractionKind::ControlFlow(_) => DagNodeKind::ControlFlow,
            InteractionKind::Synchronize => {
                if inst.op.directive() {
                    DagNodeKind::Directive
                } else {
                    match qargs.len() {
                        0 | 1 => DagNodeKind::SingleQ,
                        2 => DagNodeKind::TwoQ,
                        _ => DagNodeKind::MultiQ,
                    }
                }
            }
        };

        nodes.push(PolicyDagNodeData {
            dag_index,
            dag_indices: sabre_node.indices.clone(),
            kind,
            name: inst.op.name().to_string(),
            qargs,
            cargs,
            successors: sabre_node.successors.clone(),
            predecessors: Vec::new(),
            block_count,
        });
    }

    let mut predecessors = vec![Vec::<usize>::new(); nodes.len()];
    for (node_id, node) in nodes.iter().enumerate() {
        for &successor in &node.successors {
            predecessors[successor].push(node_id);
        }
    }
    for (node_id, preds) in predecessors.into_iter().enumerate() {
        nodes[node_id].predecessors = preds;
    }

    Ok(PolicyDagData {
        num_logical_qubits: dag.num_qubits(),
        initial_ops: sabre.initial.clone(),
        nodes,
        first_layer: sabre.first_layer.clone(),
        used_logical_qubits: collect_used_qubits(dag),
        logical_interaction_components: logical_interaction_components(dag),
    })
}

fn load_dag_via_python_qiskit(qasm3: &str) -> Result<DAGCircuit, RouterError> {
    match loads_to_dag_no_panic(qasm3) {
        Ok(dag) => Ok(dag),
        Err(primary_err) => {
            let (sanitized_qasm, changed) = sanitize_small_float_literals(qasm3);
            if !changed {
                return Err(RouterError::Parse(format!(
                    "failed converting qasm to internal DAGCircuit: {primary_err}"
                )));
            }
            loads_to_dag_no_panic(&sanitized_qasm).map_err(|retry_err| {
                RouterError::Parse(format!(
                    "failed converting qasm to internal DAGCircuit: {primary_err}; retry after sanitizing sub-1e-4 float literals failed: {retry_err}"
                ))
            })
        }
    }
}

fn loads_to_dag_no_panic(source: &str) -> Result<DAGCircuit, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let guarded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| loads_to_dag(source)));
    std::panic::set_hook(hook);
    match guarded {
        Ok(Ok(dag)) => Ok(dag),
        Ok(Err(err)) => Err(err.to_string()),
        Err(payload) => Err(format!(
            "panic in loads_to_dag: {}",
            panic_payload_to_string(payload)
        )),
    }
}

fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        return (*s).to_string();
    }
    if let Some(s) = payload.downcast_ref::<String>() {
        return s.clone();
    }
    "unknown panic payload".to_string()
}

fn sanitize_small_float_literals(qasm: &str) -> (String, bool) {
    fn is_number_char(byte: u8) -> bool {
        byte.is_ascii_digit()
            || byte == b'.'
            || byte == b'e'
            || byte == b'E'
            || byte == b'+'
            || byte == b'-'
    }

    fn is_number_start(bytes: &[u8], i: usize) -> bool {
        let c = bytes[i];
        if c.is_ascii_digit() {
            return true;
        }
        if c == b'.' {
            return i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit();
        }
        if c == b'+' || c == b'-' {
            if i + 1 >= bytes.len() {
                return false;
            }
            let n = bytes[i + 1];
            return n.is_ascii_digit() || n == b'.';
        }
        false
    }

    let bytes = qasm.as_bytes();
    let mut out = String::with_capacity(qasm.len());
    let mut i = 0usize;
    let mut changed = false;

    while i < bytes.len() {
        if !is_number_start(bytes, i) {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }

        let start = i;
        let mut j = i;
        while j < bytes.len() && is_number_char(bytes[j]) {
            j += 1;
        }

        let token = &qasm[start..j];
        let float_like = token.bytes().any(|b| b == b'.' || b == b'e' || b == b'E');
        if float_like {
            if let Ok(value) = token.parse::<f64>() {
                if value != 0.0 && value.abs() < 1.0e-4 {
                    out.push_str("0.0");
                    changed = true;
                    i = j;
                    continue;
                }
            }
        }

        out.push_str(token);
        i = j;
    }

    (out, changed)
}

fn dump_dag_via_python_qiskit(dag: &DAGCircuit) -> Result<String, RouterError> {
    dumps_from_dag(dag)
        .map_err(|e| RouterError::Routing(format!("failed exporting routed DAG to QASM3: {e}")))
}

fn restore_layout_swaps(
    from: &Layout,
    to: &Layout,
    target: &RoutingTarget,
    seed: u64,
) -> Result<Vec<(usize, usize)>, RouterError> {
    if from.logical_to_physical_map() == to.logical_to_physical_map() {
        return Ok(Vec::new());
    }
    let mapping = from
        .logical_to_physical_map()
        .iter()
        .enumerate()
        .map(|(logical, &phys)| {
            let desired = to.physical_of_logical(logical);
            (NodeIndex::new(phys), NodeIndex::new(desired))
        })
        .collect::<HbHashMap<_, _>>();

    let swaps = token_swapper(&target.graph(), mapping, Some(4), Some(seed), None)
        .map_err(|_| RouterError::Routing("token swapper could not restore layout".to_string()))?;
    Ok(swaps
        .into_iter()
        .map(|(a, b)| (a.index(), b.index()))
        .collect())
}

fn control_flow_block_layout(
    outer_layout: &Layout,
    outer_dag: &DAGCircuit,
    inst: &PackedInstruction,
    num_physical_qubits: usize,
) -> Result<Layout, RouterError> {
    let logical_count = outer_layout.logical_to_physical_map().len();
    let mut block_layout =
        Layout::from_logical_to_physical((0..num_physical_qubits).collect(), num_physical_qubits);
    for (inner, outer) in outer_dag.get_qargs(inst.qubits).iter().enumerate() {
        if outer.index() >= logical_count {
            return Err(RouterError::Routing(format!(
                "control-flow qarg {} is out of bounds for outer layout with {} logical qubits",
                outer.index(),
                logical_count
            )));
        }
        let dummy = block_layout.physical_of_logical(inner);
        let actual = outer_layout.physical_of_logical(outer.index());
        block_layout.swap_physical(dummy, actual);
    }
    Ok(block_layout)
}

fn collect_used_qubits(dag: &DAGCircuit) -> Vec<usize> {
    let mut used = Vec::<usize>::new();
    collect_used_qubits_rec(dag, &mut used, None);
    used.sort_unstable();
    used.dedup();
    used
}

fn collect_used_qubits_rec(dag: &DAGCircuit, out: &mut Vec<usize>, mapping: Option<&[usize]>) {
    for node in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node] else {
            continue;
        };
        let qargs = dag.get_qargs(inst.qubits);
        out.extend(qargs.iter().map(|q| match mapping {
            Some(map) => map[q.index()],
            None => q.index(),
        }));
        if let Some(cf) = dag.try_view_control_flow(inst) {
            let block_map = qargs
                .iter()
                .map(|q| match mapping {
                    Some(map) => map[q.index()],
                    None => q.index(),
                })
                .collect::<Vec<_>>();
            for block in cf.blocks() {
                collect_used_qubits_rec(block, out, Some(&block_map));
            }
        }
    }
}

fn logical_interaction_components(dag: &DAGCircuit) -> Vec<Vec<usize>> {
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }

    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[rb] = ra;
        }
    }

    fn add_dag(dag: &DAGCircuit, parent: &mut [usize], mapping: Option<&[usize]>) {
        for node in dag.topological_op_nodes(false) {
            let NodeType::Operation(inst) = &dag[node] else {
                continue;
            };
            let qargs = dag.get_qargs(inst.qubits);
            if let Some((first, rest)) = qargs.split_first() {
                let mapped_first = match mapping {
                    Some(map) => map[first.index()],
                    None => first.index(),
                };
                for other in rest {
                    let mapped_other = match mapping {
                        Some(map) => map[other.index()],
                        None => other.index(),
                    };
                    union(parent, mapped_first, mapped_other);
                }
            }
            if let Some(cf) = dag.try_view_control_flow(inst) {
                let block_map = qargs
                    .iter()
                    .map(|q| match mapping {
                        Some(map) => map[q.index()],
                        None => q.index(),
                    })
                    .collect::<Vec<_>>();
                for block in cf.blocks() {
                    add_dag(block, parent, Some(&block_map));
                }
            }
        }
    }

    let mut parent: Vec<usize> = (0..dag.num_qubits()).collect();
    add_dag(dag, &mut parent, None);

    let mut groups = HashMap::<usize, Vec<usize>>::new();
    for q in collect_used_qubits(dag) {
        let root = find(&mut parent, q);
        groups.entry(root).or_default().push(q);
    }
    groups
        .into_values()
        .map(|mut component| {
            component.sort_unstable();
            component
        })
        .collect()
}

fn trial_seeds(seed: u64, count: usize) -> Vec<u64> {
    if count == 0 {
        return Vec::new();
    }
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    (0..count).map(|_| rng.random::<u64>()).collect()
}

fn build_layout_trials(
    base_mapping: &[usize],
    target: &RoutingTarget,
    layout_trials: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let mut out = Vec::<Vec<usize>>::new();
    out.push(base_mapping.to_vec());
    if layout_trials <= 1 {
        return out;
    }

    let logical_count = base_mapping.len();
    let num_components = target.connected_components().len();
    let logical_component_of_base = base_mapping
        .iter()
        .map(|&physical| target.component_of_physical(physical))
        .collect::<Vec<_>>();
    let mut logicals_by_component = vec![Vec::<usize>::new(); num_components];
    for (logical, &component) in logical_component_of_base.iter().enumerate() {
        logicals_by_component[component].push(logical);
    }

    let mut rng = Pcg64Mcg::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15);
    for _ in 1..layout_trials {
        let mut trial = base_mapping.to_vec();
        let mut touched = vec![false; target.num_qubits];
        for &physical in &trial {
            touched[physical] = true;
        }
        let mut free_by_component = vec![Vec::<usize>::new(); num_components];
        for physical in 0..target.num_qubits {
            if touched[physical] {
                continue;
            }
            let component = target.component_of_physical(physical);
            free_by_component[component].push(physical);
        }

        for logical in 0..logical_count {
            let component = logical_component_of_base[logical];
            let same_component_logicals = &logicals_by_component[component];
            if same_component_logicals.len() > 1 && rng.random::<f64>() < 0.35 {
                let other = same_component_logicals[rng.random_range(0..same_component_logicals.len())];
                trial.swap(logical, other);
            } else if !free_by_component[component].is_empty() && rng.random::<f64>() < 0.25 {
                let idx = rng.random_range(0..free_by_component[component].len());
                let new_physical = free_by_component[component].swap_remove(idx);
                free_by_component[component].push(trial[logical]);
                trial[logical] = new_physical;
            }
        }
        out.push(trial);
    }
    out
}

fn compute_depth_from_dag(dag: &DAGCircuit) -> usize {
    let mut depths = vec![0usize; dag.num_qubits()];
    compute_depth_rec(dag, &mut depths);
    depths.into_iter().max().unwrap_or(0)
}

fn compute_depth_rec(dag: &DAGCircuit, depths: &mut [usize]) {
    for node in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node] else {
            continue;
        };
        if let Some(cf) = dag.try_view_control_flow(inst) {
            let mut branch_end = depths.iter().copied().max().unwrap_or(0);
            for block in cf.blocks() {
                let mut block_depths = depths.to_vec();
                compute_depth_rec(block, &mut block_depths);
                branch_end = branch_end.max(block_depths.into_iter().max().unwrap_or(branch_end));
            }
            for q in dag.get_qargs(inst.qubits) {
                let idx = q.index();
                depths[idx] = depths[idx].max(branch_end);
            }
            continue;
        }

        let qargs = dag.get_qargs(inst.qubits);
        match qargs.len() {
            0 => {}
            1 => {
                let q = qargs[0].index();
                depths[q] += 1;
            }
            _ => {
                let next = qargs
                    .iter()
                    .map(|q| depths[q.index()])
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                for q in qargs {
                    depths[q.index()] = next;
                }
            }
        }
    }
}

fn count_twoq_ops(dag: &DAGCircuit) -> usize {
    let mut total = 0usize;
    for node in dag.topological_op_nodes(false) {
        let NodeType::Operation(inst) = &dag[node] else {
            continue;
        };
        if let Some(cf) = dag.try_view_control_flow(inst) {
            for block in cf.blocks() {
                total += count_twoq_ops(block);
            }
            continue;
        }
        let is_swap = matches!(inst.op.try_standard_gate(), Some(StandardGate::Swap));
        if inst.op.num_qubits() == 2 && !is_swap {
            total += 1;
        }
    }
    total
}
