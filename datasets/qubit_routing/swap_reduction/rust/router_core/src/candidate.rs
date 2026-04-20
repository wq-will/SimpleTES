use std::collections::{HashMap, HashSet, VecDeque};
use std::ops::Range;

use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use rustworkx_core::petgraph::graph::NodeIndex;

use crate::engine::dense_layout;
use crate::engine::routing_target::RoutingTarget;
use crate::engine::sabre::layout::Layout;
use crate::engine::RouterError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DagNodeKind {
    Directive,
    SingleQ,
    TwoQ,
    MultiQ,
    ControlFlow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemainingNodeState {
    Pending,
    Ready,
    Executed,
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyDagNodeData {
    pub(crate) dag_index: NodeIndex,
    pub(crate) dag_indices: Vec<NodeIndex>,
    pub(crate) kind: DagNodeKind,
    pub(crate) name: String,
    pub(crate) qargs: Vec<usize>,
    pub(crate) cargs: Vec<usize>,
    pub(crate) successors: Vec<usize>,
    pub(crate) predecessors: Vec<usize>,
    pub(crate) block_count: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct PolicyDagData {
    pub(crate) num_logical_qubits: usize,
    pub(crate) initial_ops: Vec<NodeIndex>,
    pub(crate) nodes: Vec<PolicyDagNodeData>,
    pub(crate) first_layer: Vec<usize>,
    pub(crate) used_logical_qubits: Vec<usize>,
    pub(crate) logical_interaction_components: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Copy)]
pub struct DagNodeView<'a> {
    node: &'a PolicyDagNodeData,
}

impl<'a> DagNodeView<'a> {
    pub(crate) fn new(node: &'a PolicyDagNodeData) -> Self {
        Self { node }
    }

    pub fn kind(&self) -> DagNodeKind {
        self.node.kind
    }

    pub fn name(&self) -> &str {
        &self.node.name
    }

    pub fn qargs(&self) -> &[usize] {
        &self.node.qargs
    }

    pub fn cargs(&self) -> &[usize] {
        &self.node.cargs
    }

    pub fn successors(&self) -> &[usize] {
        &self.node.successors
    }

    pub fn predecessors(&self) -> &[usize] {
        &self.node.predecessors
    }

    pub fn block_count(&self) -> usize {
        self.node.block_count
    }

    pub fn is_two_qubit(&self) -> bool {
        self.node.kind == DagNodeKind::TwoQ && self.node.qargs.len() == 2
    }

    pub fn two_qubit_pair(&self) -> Option<(usize, usize)> {
        if self.is_two_qubit() {
            Some((self.node.qargs[0], self.node.qargs[1]))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CircuitDagView<'a> {
    data: &'a PolicyDagData,
}

impl<'a> CircuitDagView<'a> {
    pub(crate) fn new(data: &'a PolicyDagData) -> Self {
        Self { data }
    }

    pub fn num_logical_qubits(&self) -> usize {
        self.data.num_logical_qubits
    }

    pub fn node_count(&self) -> usize {
        self.data.nodes.len()
    }

    pub fn node_ids(&self) -> Range<usize> {
        0..self.data.nodes.len()
    }

    pub fn node(&self, node_id: usize) -> DagNodeView<'a> {
        DagNodeView::new(&self.data.nodes[node_id])
    }

    pub fn first_layer_node_ids(&self) -> &[usize] {
        &self.data.first_layer
    }

    pub fn used_logical_qubits(&self) -> &[usize] {
        &self.data.used_logical_qubits
    }

    pub fn logical_interaction_components(&self) -> &[Vec<usize>] {
        &self.data.logical_interaction_components
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TopologyView<'a> {
    target: &'a RoutingTarget,
}

impl<'a> TopologyView<'a> {
    pub(crate) fn new(target: &'a RoutingTarget) -> Self {
        Self { target }
    }

    pub fn num_qubits(&self) -> usize {
        self.target.num_qubits
    }

    pub fn num_edges(&self) -> usize {
        self.target.edges().len()
    }

    pub fn edges(&self) -> &[[usize; 2]] {
        self.target.edges()
    }

    pub fn neighbors(&self, qubit: usize) -> &[usize] {
        self.target.neighbors(qubit)
    }

    pub fn degree(&self, qubit: usize) -> usize {
        self.target.degree(qubit)
    }

    pub fn max_degree(&self) -> usize {
        self.target.max_degree()
    }

    pub fn distance(&self, a: usize, b: usize) -> u32 {
        self.target.distance(a, b)
    }

    pub fn distances(&self) -> &[Vec<u32>] {
        self.target.distances()
    }

    pub fn shortest_path(&self, a: usize, b: usize) -> Vec<usize> {
        self.target.shortest_path(a, b)
    }

    pub fn connected_components(&self) -> &[Vec<usize>] {
        self.target.connected_components()
    }

    pub fn component_of_physical(&self, physical: usize) -> usize {
        self.target.component_of_physical(physical)
    }

    pub fn is_adjacent(&self, a: usize, b: usize) -> bool {
        self.target.is_adjacent(a, b)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LayoutView<'a> {
    layout: &'a Layout,
}

impl<'a> LayoutView<'a> {
    pub(crate) fn new(layout: &'a Layout) -> Self {
        Self { layout }
    }

    pub fn num_logical(&self) -> usize {
        self.layout.logical_to_physical_map().len()
    }

    pub fn num_physical(&self) -> usize {
        self.layout.num_physical()
    }

    pub fn physical_of_logical(&self, logical: usize) -> usize {
        self.layout.physical_of_logical(logical)
    }

    pub fn logical_of_physical(&self, physical: usize) -> Option<usize> {
        self.layout.logical_of_physical(physical)
    }

    pub fn logical_to_physical(&self) -> &[usize] {
        self.layout.logical_to_physical_map()
    }

    pub fn physical_to_logical(&self) -> &[Option<usize>] {
        self.layout.physical_to_logical_map()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FrontLayerView<'a> {
    circuit: CircuitDagView<'a>,
    layout: LayoutView<'a>,
    node_ids: &'a [usize],
    physical_pairs: &'a [[usize; 2]],
}

impl<'a> FrontLayerView<'a> {
    pub(crate) fn new(
        circuit: CircuitDagView<'a>,
        layout: LayoutView<'a>,
        node_ids: &'a [usize],
        physical_pairs: &'a [[usize; 2]],
    ) -> Self {
        Self {
            circuit,
            layout,
            node_ids,
            physical_pairs,
        }
    }

    pub fn len(&self) -> usize {
        self.node_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_ids.is_empty()
    }

    pub fn node_ids(&self) -> &[usize] {
        self.node_ids
    }

    pub fn logical_pair(&self, node_id: usize) -> Option<(usize, usize)> {
        self.circuit.node(node_id).two_qubit_pair()
    }

    pub fn physical_pair(&self, node_id: usize) -> Option<(usize, usize)> {
        let idx = self.node_ids.iter().position(|&id| id == node_id)?;
        let pair = self.physical_pairs[idx];
        Some((pair[0], pair[1]))
    }

    pub fn active_logical_qubits(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for &node_id in self.node_ids {
            if let Some((a, b)) = self.logical_pair(node_id) {
                if seen.insert(a) {
                    out.push(a);
                }
                if seen.insert(b) {
                    out.push(b);
                }
            }
        }
        out
    }

    pub fn active_physical_qubits(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for pair in self.physical_pairs {
            out.push(pair[0]);
            out.push(pair[1]);
        }
        out
    }

    pub fn physical_pairs(&self) -> &'a [[usize; 2]] {
        self.physical_pairs
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RemainingDagView<'a> {
    circuit: CircuitDagView<'a>,
    state: &'a [RemainingNodeState],
    remaining_predecessors: &'a [usize],
    ready_node_ids: &'a [usize],
    pending_node_ids: &'a [usize],
    remaining_two_qubit_node_ids: &'a [usize],
    front_layer_node_ids: &'a [usize],
}

impl<'a> RemainingDagView<'a> {
    pub(crate) fn new(
        circuit: CircuitDagView<'a>,
        state: &'a [RemainingNodeState],
        remaining_predecessors: &'a [usize],
        ready_node_ids: &'a [usize],
        pending_node_ids: &'a [usize],
        remaining_two_qubit_node_ids: &'a [usize],
        front_layer_node_ids: &'a [usize],
    ) -> Self {
        Self {
            circuit,
            state,
            remaining_predecessors,
            ready_node_ids,
            pending_node_ids,
            remaining_two_qubit_node_ids,
            front_layer_node_ids,
        }
    }

    pub fn circuit(&self) -> CircuitDagView<'a> {
        self.circuit
    }

    pub fn state(&self, node_id: usize) -> RemainingNodeState {
        self.state[node_id]
    }

    pub fn is_executed(&self, node_id: usize) -> bool {
        self.state(node_id) == RemainingNodeState::Executed
    }

    pub fn remaining_predecessor_count(&self, node_id: usize) -> usize {
        self.remaining_predecessors[node_id]
    }

    pub fn remaining_predecessor_counts(&self) -> &[usize] {
        self.remaining_predecessors
    }

    pub fn ready_node_ids(&self) -> &[usize] {
        self.ready_node_ids
    }

    pub fn pending_node_ids(&self) -> &[usize] {
        self.pending_node_ids
    }

    pub fn unexecuted_node_ids(&self) -> &[usize] {
        self.pending_node_ids
    }

    pub fn remaining_two_qubit_node_ids(&self) -> &[usize] {
        self.remaining_two_qubit_node_ids
    }

    pub fn front_layer_node_ids(&self) -> &[usize] {
        self.front_layer_node_ids
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InitialLayoutContext<'a> {
    topology: TopologyView<'a>,
    circuit: CircuitDagView<'a>,
}

impl<'a> InitialLayoutContext<'a> {
    pub(crate) fn new(topology: TopologyView<'a>, circuit: CircuitDagView<'a>) -> Self {
        Self { topology, circuit }
    }

    pub fn topology(&self) -> TopologyView<'a> {
        self.topology
    }

    pub fn circuit(&self) -> CircuitDagView<'a> {
        self.circuit
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SwapSelectionContext<'a> {
    topology: TopologyView<'a>,
    layout: LayoutView<'a>,
    circuit: CircuitDagView<'a>,
    remaining: RemainingDagView<'a>,
    front_layer: FrontLayerView<'a>,
    precomputed_extended_set_logical_pairs: Option<&'a [[usize; 2]]>,
    swaps_since_progress: usize,
    last_applied_swap: Option<(usize, usize)>,
}

impl<'a> SwapSelectionContext<'a> {
    pub(crate) fn new(
        topology: TopologyView<'a>,
        layout: LayoutView<'a>,
        circuit: CircuitDagView<'a>,
        remaining: RemainingDagView<'a>,
        front_layer: FrontLayerView<'a>,
        swaps_since_progress: usize,
        last_applied_swap: Option<(usize, usize)>,
    ) -> Self {
        Self {
            topology,
            layout,
            circuit,
            remaining,
            front_layer,
            precomputed_extended_set_logical_pairs: None,
            swaps_since_progress,
            last_applied_swap,
        }
    }

    pub(crate) fn new_with_precomputed_extended_set(
        topology: TopologyView<'a>,
        layout: LayoutView<'a>,
        circuit: CircuitDagView<'a>,
        remaining: RemainingDagView<'a>,
        front_layer: FrontLayerView<'a>,
        precomputed_extended_set_logical_pairs: &'a [[usize; 2]],
        swaps_since_progress: usize,
        last_applied_swap: Option<(usize, usize)>,
    ) -> Self {
        Self {
            topology,
            layout,
            circuit,
            remaining,
            front_layer,
            precomputed_extended_set_logical_pairs: Some(precomputed_extended_set_logical_pairs),
            swaps_since_progress,
            last_applied_swap,
        }
    }

    pub fn topology(&self) -> TopologyView<'a> {
        self.topology
    }

    pub fn layout(&self) -> LayoutView<'a> {
        self.layout
    }

    pub fn circuit(&self) -> CircuitDagView<'a> {
        self.circuit
    }

    pub fn remaining(&self) -> RemainingDagView<'a> {
        self.remaining
    }

    pub fn front_layer(&self) -> FrontLayerView<'a> {
        self.front_layer
    }

    pub fn precomputed_extended_set_logical_pairs(&self) -> &'a [[usize; 2]] {
        self.precomputed_extended_set_logical_pairs.unwrap_or(&[])
    }

    pub fn swaps_since_progress(&self) -> usize {
        self.swaps_since_progress
    }

    pub fn last_applied_swap(&self) -> Option<(usize, usize)> {
        self.last_applied_swap
    }
}

#[derive(Debug, Clone)]
pub struct RngState {
    rng: Pcg64Mcg,
}

impl RngState {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg64Mcg::seed_from_u64(seed),
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.rng.random::<u64>()
    }

    pub fn gen_index(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            0
        } else {
            self.rng.random_range(0..upper)
        }
    }

    pub fn gen_f64(&mut self) -> f64 {
        self.rng.random::<f64>()
    }

    pub fn shuffle_slice<T>(&mut self, values: &mut [T]) {
        if values.len() <= 1 {
            return;
        }
        for i in (1..values.len()).rev() {
            let j = self.rng.random_range(0..=i);
            values.swap(i, j);
        }
    }
}

pub trait Policy: Clone {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError>;

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)>;
}

// EVOLVE-BLOCK-START
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

#[derive(Debug, Clone, Default)]
struct FrontLayerScores {
    nodes: Vec<[usize; 2]>,
    qubits: Vec<Option<(usize, usize)>>,
}

impl FrontLayerScores {
    fn from_ctx(ctx: &SwapSelectionContext<'_>) -> Self {
        let mut out = Self {
            nodes: Vec::new(),
            qubits: vec![None; ctx.topology().num_qubits()],
        };
        for pair in ctx.front_layer().physical_pairs() {
            let [a, b] = *pair;
            let index = out.nodes.len();
            out.nodes.push([a, b]);
            out.qubits[a] = Some((index, b));
            out.qubits[b] = Some((index, a));
        }
        out
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.iter().flatten()
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.nodes
            .iter()
            .map(|pair| topology.distance(pair[0], pair[1]) as f64)
            .sum()
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut delta = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            delta += (topology.distance(b, c) as f64) - (topology.distance(a, c) as f64);
        }
        if let Some((_, c)) = self.qubits[b] {
            delta += (topology.distance(a, c) as f64) - (topology.distance(b, c) as f64);
        }
        delta
    }
}

#[derive(Debug, Clone)]
struct ExtendedSetScores {
    qubits: Vec<Vec<usize>>,
    len: usize,
}

impl ExtendedSetScores {
    fn new(num_qubits: usize) -> Self {
        Self {
            qubits: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    fn push(&mut self, a: usize, b: usize) {
        self.qubits[a].push(b);
        self.qubits[b].push(a);
        self.len += 1;
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn total_score(&self, topology: TopologyView<'_>) -> f64 {
        self.qubits
            .iter()
            .enumerate()
            .flat_map(|(a, others)| {
                others
                    .iter()
                    .map(move |b| topology.distance(a, *b) as f64)
            })
            .sum::<f64>()
            * 0.5
    }

    fn score_delta(&self, swap: (usize, usize), topology: TopologyView<'_>) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b {
                continue;
            }
            total += (topology.distance(b, *other) as f64) - (topology.distance(a, *other) as f64);
        }
        for other in &self.qubits[b] {
            if *other == a {
                continue;
            }
            total += (topology.distance(a, *other) as f64) - (topology.distance(b, *other) as f64);
        }
        total
    }
}

fn build_extended_set(ctx: &SwapSelectionContext<'_>, max_size: usize) -> ExtendedSetScores {
    let mut out = ExtendedSetScores::new(ctx.topology().num_qubits());
    if max_size == 0 {
        return out;
    }

    let precomputed = ctx.precomputed_extended_set_logical_pairs();
    if !precomputed.is_empty() {
        for pair in precomputed.iter().take(max_size) {
            out.push(
                ctx.layout().physical_of_logical(pair[0]),
                ctx.layout().physical_of_logical(pair[1]),
            );
        }
        return out;
    }

    let mut required_predecessors = ctx.remaining().remaining_predecessor_counts().to_vec();
    let mut to_visit = ctx.front_layer().node_ids().to_vec();
    let mut decremented = Vec::<(usize, usize)>::new();
    let mut i = 0usize;
    while i < to_visit.len() && out.len() < max_size {
        let node_id = to_visit[i];
        for &successor in ctx.circuit().node(node_id).successors() {
            if let Some((_, amount)) = decremented.iter_mut().find(|(idx, _)| *idx == successor) {
                *amount += 1;
            } else {
                decremented.push((successor, 1));
            }
            required_predecessors[successor] -= 1;
            if required_predecessors[successor] == 0 {
                if let Some((a, b)) = ctx.circuit().node(successor).two_qubit_pair() {
                    out.push(ctx.layout().physical_of_logical(a), ctx.layout().physical_of_logical(b));
                }
                to_visit.push(successor);
            }
        }
        i += 1;
    }

    out
}

fn enumerate_candidate_swaps(
    topology: TopologyView<'_>,
    front_layer: &FrontLayerScores,
) -> Vec<(usize, usize)> {
    let mut out = Vec::<(usize, usize)>::new();

    for &phys in front_layer.iter_active() {
        for &neighbor in topology.neighbors(phys) {
            if neighbor > phys || !front_layer.is_active(neighbor) {
                out.push((phys, neighbor));
            }
        }
    }

    out
}

fn choose_dense_layout_subset(
    topology: TopologyView<'_>,
    logical_component_size: usize,
    target_component: &[usize],
) -> Result<Vec<usize>, RouterError> {
    if logical_component_size > target_component.len() {
        return Err(RouterError::Routing(format!(
            "logical component size {logical_component_size} exceeds target component size {}",
            target_component.len()
        )));
    }
    if logical_component_size == target_component.len() {
        return Ok(target_component.to_vec());
    }

    let local_index = target_component
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<HashMap<usize, usize>>();
    let mut local_adj = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    for &global_a in target_component {
        let a = local_index[&global_a];
        for &global_b in topology.neighbors(global_a) {
            if let Some(&b) = local_index.get(&global_b) {
                local_adj[[a, b]] = 1.0;
            }
        }
    }
    let error_matrix = Array2::<f64>::zeros((target_component.len(), target_component.len()));
    let [_, _, best_map] = dense_layout::best_subset(
        logical_component_size,
        local_adj.view(),
        0,
        0,
        false,
        true,
        error_matrix.view(),
    );
    let chosen = best_map
        .into_iter()
        .take(logical_component_size)
        .map(|local| target_component[local])
        .collect::<Vec<_>>();
    ensure_connected_subset(topology, &chosen)?;
    Ok(chosen)
}

fn ensure_connected_subset(topology: TopologyView<'_>, subset: &[usize]) -> Result<(), RouterError> {
    if subset.is_empty() {
        return Ok(());
    }
    let set = subset.iter().copied().collect::<HashSet<_>>();
    let mut seen = HashSet::<usize>::new();
    let mut queue = VecDeque::<usize>::new();
    queue.push_back(subset[0]);
    seen.insert(subset[0]);

    while let Some(node) = queue.pop_front() {
        for &next in topology.neighbors(node) {
            if set.contains(&next) && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }

    if seen.len() != set.len() {
        return Err(RouterError::Routing(
            "selected layout subset is not connected".to_string(),
        ));
    }
    Ok(())
}

fn assign_components_to_target(
    logical_components: &[Vec<usize>],
    target_components: &[Vec<usize>],
) -> Result<Vec<(Vec<usize>, usize)>, RouterError> {
    if logical_components.is_empty() {
        return Ok(Vec::new());
    }

    let mut logical_sorted = logical_components.to_vec();
    logical_sorted.sort_by_key(|component| std::cmp::Reverse(component.len()));

    let mut target_sorted = target_components
        .iter()
        .enumerate()
        .map(|(idx, component)| (idx, component.len()))
        .collect::<Vec<_>>();
    target_sorted.sort_by_key(|(_, size)| std::cmp::Reverse(*size));

    let mut free_capacity = target_sorted
        .iter()
        .map(|(idx, size)| (*idx, *size))
        .collect::<HashMap<usize, usize>>();
    let mut assignments = Vec::<(Vec<usize>, usize)>::new();

    for logical in logical_sorted {
        let size = logical.len();
        let mut chosen = None;
        for (target_idx, _) in &target_sorted {
            let cap = free_capacity.get(target_idx).copied().unwrap_or(0);
            if cap >= size {
                chosen = Some(*target_idx);
                break;
            }
        }
        let Some(target_idx) = chosen else {
            return Err(RouterError::Routing(format!(
                "logical component of size {size} cannot fit any target component"
            )));
        };
        *free_capacity
            .get_mut(&target_idx)
            .expect("selected target component must exist") -= size;
        assignments.push((logical, target_idx));
    }
    Ok(assignments)
}

fn choose_disjoint_aware_layout(ctx: &InitialLayoutContext<'_>) -> Result<Vec<usize>, RouterError> {
    let circuit = ctx.circuit();
    let topology = ctx.topology();
    let num_logical = circuit.num_logical_qubits();
    let used = circuit.used_logical_qubits();
    if used.is_empty() {
        return Ok((0..num_logical).collect());
    }

    let logical_components = circuit.logical_interaction_components();
    let target_components = topology.connected_components();
    if target_components.is_empty() {
        return Err(RouterError::Routing(
            "topology has no connected components".to_string(),
        ));
    }

    let assignments = assign_components_to_target(logical_components, target_components)?;
    let mut mapping = vec![usize::MAX; num_logical];
    let mut used_physical = vec![false; topology.num_qubits()];

    for (logical_component, target_component_idx) in assignments {
        if logical_component.is_empty() {
            continue;
        }
        let target_component = &target_components[target_component_idx];
        let local = choose_dense_layout_subset(topology, logical_component.len(), target_component)?;
        for (logical, physical) in logical_component.iter().zip(local) {
            mapping[*logical] = physical;
            used_physical[physical] = true;
        }
    }

    let mut free_physical = (0..topology.num_qubits()).filter(|q| !used_physical[*q]);
    for slot in &mut mapping {
        if *slot == usize::MAX {
            *slot = free_physical.next().ok_or_else(|| {
                RouterError::Routing("not enough physical qubits to complete layout".to_string())
            })?;
        }
    }
    Ok(mapping)
}

#[derive(Debug, Clone)]
pub struct CandidatePolicy {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub lookahead_size: usize,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    pub decay_increment: f64,
    pub decay_reset: usize,
    pub best_epsilon: f64,
    decay_state: Vec<f64>,
}

impl Default for CandidatePolicy {
    fn default() -> Self {
        Self {
            basic_weight: 1.0,
            lookahead_weight: 0.5,
            lookahead_size: 20,
            set_scaling: SetScaling::Size,
            use_decay: true,
            decay_increment: 0.001,
            decay_reset: 5,
            best_epsilon: 1e-10,
            decay_state: Vec::new(),
        }
    }
}

impl CandidatePolicy {
    fn refresh_decay_state(&mut self, ctx: &SwapSelectionContext<'_>) {
        if !self.use_decay {
            return;
        }

        let num_qubits = ctx.topology().num_qubits();
        if self.decay_state.len() != num_qubits {
            self.decay_state = vec![1.0; num_qubits];
        }

        if ctx.swaps_since_progress() == 0 {
            self.decay_state.fill(1.0);
            return;
        }

        let Some((a, b)) = ctx.last_applied_swap() else {
            return;
        };
        let reset = self.decay_reset.max(1);
        if ctx.swaps_since_progress() % reset == 0 {
            self.decay_state.fill(1.0);
        } else {
            self.decay_state[a] += self.decay_increment;
            self.decay_state[b] += self.decay_increment;
        }
    }
}

impl Policy for CandidatePolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, RouterError> {
        self.decay_state.clear();
        choose_disjoint_aware_layout(ctx)
    }

    fn choose_best_swap(
        &mut self,
        ctx: &SwapSelectionContext<'_>,
        rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        self.refresh_decay_state(ctx);

        let front_layer = FrontLayerScores::from_ctx(ctx);
        let candidates = enumerate_candidate_swaps(ctx.topology(), &front_layer);
        if candidates.is_empty() {
            return None;
        }

        let extended_set = build_extended_set(ctx, self.lookahead_size);

        let scale = |weight: f64, size: usize, scaling: SetScaling| -> f64 {
            match scaling {
                SetScaling::Constant => weight,
                SetScaling::Size => {
                    if size == 0 {
                        0.0
                    } else {
                        weight / (size as f64)
                    }
                }
            }
        };

        let basic_weight = scale(self.basic_weight, front_layer.len(), self.set_scaling);
        let lookahead_weight = scale(
            self.lookahead_weight,
            extended_set.len(),
            self.set_scaling,
        );

        let mut swap_scores = candidates
            .iter()
            .copied()
            .map(|swap| (swap, 0.0))
            .collect::<Vec<_>>();

        let mut absolute_score = 0.0;
        absolute_score += basic_weight * front_layer.total_score(ctx.topology());
        for (swap, score) in &mut swap_scores {
            *score += basic_weight * front_layer.score_delta(*swap, ctx.topology());
        }

        if !extended_set.is_empty() && self.lookahead_weight != 0.0 {
            absolute_score += lookahead_weight * extended_set.total_score(ctx.topology());
            for (swap, score) in &mut swap_scores {
                *score += lookahead_weight * extended_set.score_delta(*swap, ctx.topology());
            }
        }

        if self.use_decay {
            for (swap, score) in &mut swap_scores {
                *score =
                    (absolute_score + *score) * self.decay_state[swap.0].max(self.decay_state[swap.1]);
            }
        } else {
            for (_, score) in &mut swap_scores {
                *score = absolute_score + *score;
            }
        }

        let mut min_score = f64::INFINITY;
        let mut best_swaps = Vec::<(usize, usize)>::new();
        for (swap, score) in &swap_scores {
            if *score + self.best_epsilon < min_score {
                min_score = *score;
                best_swaps.clear();
                best_swaps.push(*swap);
                continue;
            }
            if (*score - min_score).abs() <= self.best_epsilon {
                best_swaps.push(*swap);
            }
        }

        Some(best_swaps[rng.gen_index(best_swaps.len())])
    }
}
// EVOLVE-BLOCK-END
