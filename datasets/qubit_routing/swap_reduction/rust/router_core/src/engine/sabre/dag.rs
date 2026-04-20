use std::collections::HashMap;

use qiskit_circuit::dag_circuit::{DAGCircuit, NodeType, Wire};
use qiskit_circuit::operations::{ControlFlowView, Operation};
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{Qubit, VirtualQubit};
use rustworkx_core::petgraph::prelude::*;

use crate::engine::RouterError;

#[derive(Clone, Debug)]
pub enum InteractionKind {
    Synchronize,
    TwoQ([VirtualQubit; 2]),
    ControlFlow(Box<[(SabreDAG, DAGCircuit)]>),
}

impl InteractionKind {
    fn from_control_flow(cf: ControlFlowView<DAGCircuit>) -> Result<Self, RouterError> {
        let blocks = cf
            .blocks()
            .into_iter()
            .map(|dag| Ok((SabreDAG::from_dag(dag)?, dag.clone())))
            .collect::<Result<Box<[_]>, RouterError>>()?;
        Ok(Self::ControlFlow(blocks))
    }

    fn from_op(op: &PackedOperation, qargs: &[Qubit]) -> Self {
        if op.directive() {
            return Self::Synchronize;
        }
        match qargs {
            &[left, right] => Self::TwoQ([VirtualQubit::new(left.0), VirtualQubit::new(right.0)]),
            _ => Self::Synchronize,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SabreNode {
    pub indices: Vec<NodeIndex>,
    pub kind: InteractionKind,
    pub successors: Vec<usize>,
    pub predecessors: usize,
}

impl SabreNode {
    fn new(initial: NodeIndex, kind: InteractionKind) -> Self {
        Self {
            indices: vec![initial],
            kind,
            successors: Vec::new(),
            predecessors: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SabreDAG {
    pub initial: Vec<NodeIndex>,
    pub nodes: Vec<SabreNode>,
    pub first_layer: Vec<usize>,
}

impl SabreDAG {
    pub fn from_dag(dag: &DAGCircuit) -> Result<Self, RouterError> {
        let mut initial = Vec::<NodeIndex>::new();
        let mut nodes = Vec::<SabreNode>::new();
        let mut wire_pos: HashMap<Wire, usize> = HashMap::with_capacity(dag.width());
        let mut first_layer = Vec::<usize>::new();

        enum Predecessors {
            AllUnmapped,
            Single(usize),
            Multiple,
        }

        let same_twoq_pair = |left: &[VirtualQubit; 2], right: &[VirtualQubit; 2]| -> bool {
            let mut l = [left[0].index(), left[1].index()];
            let mut r = [right[0].index(), right[1].index()];
            l.sort_unstable();
            r.sort_unstable();
            l == r
        };

        let predecessors =
            |dag_node: NodeIndex, sabre_pos: &HashMap<Wire, usize>| -> Predecessors {
                let mut edges = dag.dag().edges_directed(dag_node, Direction::Incoming);
                let Some(first) = edges.next() else {
                    return Predecessors::AllUnmapped;
                };
                let single = sabre_pos.get(first.weight()).copied();
                for edge in edges {
                    if single != sabre_pos.get(edge.weight()).copied() {
                        return Predecessors::Multiple;
                    }
                }
                single.map_or(Predecessors::AllUnmapped, Predecessors::Single)
            };

        fn add_edge(nodes: &mut [SabreNode], parent: usize, child: usize) {
            if parent == child {
                return;
            }
            nodes[parent].successors.push(child);
            nodes[child].predecessors += 1;
        }

        for dag_node in dag.topological_op_nodes(false) {
            let NodeType::Operation(inst) = &dag[dag_node] else {
                continue;
            };
            let kind = if let Some(cf) = dag.try_view_control_flow(inst) {
                InteractionKind::from_control_flow(cf)?
            } else {
                InteractionKind::from_op(&inst.op, dag.get_qargs(inst.qubits))
            };

            match predecessors(dag_node, &wire_pos) {
                Predecessors::AllUnmapped => match kind {
                    InteractionKind::Synchronize => initial.push(dag_node),
                    kind => {
                        let node_id = nodes.len();
                        nodes.push(SabreNode::new(dag_node, kind));
                        first_layer.push(node_id);
                        for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                            wire_pos.insert(*edge.weight(), node_id);
                        }
                    }
                },
                Predecessors::Multiple => {
                    let node_id = nodes.len();
                    nodes.push(SabreNode::new(dag_node, kind));
                    for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                        if let Some(parent) = wire_pos.insert(*edge.weight(), node_id) {
                            add_edge(&mut nodes, parent, node_id);
                        }
                    }
                }
                Predecessors::Single(prev) => {
                    let fold = matches!(kind, InteractionKind::Synchronize)
                        || match (&nodes[prev].kind, &kind) {
                            (InteractionKind::TwoQ(prev_pair), InteractionKind::TwoQ(cur_pair)) => {
                                same_twoq_pair(prev_pair, cur_pair)
                            }
                            _ => false,
                        };
                    if fold {
                        nodes[prev].indices.push(dag_node);
                    } else {
                        let node_id = nodes.len();
                        nodes.push(SabreNode::new(dag_node, kind));
                        add_edge(&mut nodes, prev, node_id);
                        for edge in dag.dag().edges_directed(dag_node, Direction::Incoming) {
                            wire_pos.insert(*edge.weight(), node_id);
                        }
                    }
                }
            }
        }

        Ok(Self {
            initial,
            nodes,
            first_layer,
        })
    }
}
