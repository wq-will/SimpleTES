use std::fs;
use std::path::Path;

use crate::candidate::{
    InitialLayoutContext, LayoutView, Policy, RngState, SwapSelectionContext, TopologyView,
};
use crate::engine::sabre::layout::Layout;
use crate::{route_qasm3_with_policy, CandidatePolicy, RouteOptions, RoutingTarget};

#[derive(Clone, Default)]
struct InvalidInitialLayoutPolicy;

impl Policy for InvalidInitialLayoutPolicy {
    fn choose_best_initial_layout(
        &mut self,
        _ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, crate::RouterError> {
        Ok(vec![0, 0])
    }

    fn choose_best_swap(
        &mut self,
        _ctx: &SwapSelectionContext<'_>,
        _rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        None
    }
}

#[derive(Clone, Default)]
struct FallbackOnlyPolicy;

impl Policy for FallbackOnlyPolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, crate::RouterError> {
        Ok((0..ctx.circuit().num_logical_qubits()).collect())
    }

    fn choose_best_swap(
        &mut self,
        _ctx: &SwapSelectionContext<'_>,
        _rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        None
    }
}

#[derive(Clone, Default)]
struct InvalidSwapPolicy;

impl Policy for InvalidSwapPolicy {
    fn choose_best_initial_layout(
        &mut self,
        ctx: &InitialLayoutContext<'_>,
        _rng: &mut RngState,
    ) -> Result<Vec<usize>, crate::RouterError> {
        Ok((0..ctx.circuit().num_logical_qubits()).collect())
    }

    fn choose_best_swap(
        &mut self,
        _ctx: &SwapSelectionContext<'_>,
        _rng: &mut RngState,
    ) -> Option<(usize, usize)> {
        Some((0, 2))
    }
}

fn extract_evolve_block_body(source: &str) -> &str {
    let start_marker = "EVOLVE-BLOCK-START";
    let end_marker = "EVOLVE-BLOCK-END";

    let start_idx = source
        .find(start_marker)
        .unwrap_or_else(|| panic!("missing {start_marker} marker"));
    let body_start = source[start_idx..]
        .find('\n')
        .map(|offset| start_idx + offset + 1)
        .unwrap_or_else(|| panic!("missing newline after {start_marker} marker"));
    let end_idx = source[body_start..]
        .find(end_marker)
        .map(|offset| body_start + offset)
        .unwrap_or_else(|| panic!("missing {end_marker} marker"));

    source[body_start..end_idx].trim()
}

#[test]
fn candidate_scaffold_contains_init_program_evolve_block() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidate = fs::read_to_string(root.join("src/candidate.rs")).unwrap();
    let init_program = fs::read_to_string(root.join("../../init_program.rs")).unwrap();
    assert_eq!(
        extract_evolve_block_body(&candidate),
        extract_evolve_block_body(&init_program),
    );
    assert!(candidate.len() > init_program.len());
}

#[test]
fn topology_view_reports_components_and_edges() {
    let target = RoutingTarget::new(5, &[[0, 1], [1, 2], [3, 4]]).unwrap();
    let view = TopologyView::new(&target);

    assert_eq!(view.num_qubits(), 5);
    assert_eq!(view.num_edges(), 3);
    assert_eq!(view.edges(), &[[0, 1], [1, 2], [3, 4]]);
    assert_eq!(view.degree(1), 2);
    assert_eq!(view.max_degree(), 2);
    assert_eq!(view.connected_components(), &[vec![0, 1, 2], vec![3, 4]]);
    assert_eq!(view.component_of_physical(4), 1);
    assert_eq!(view.shortest_path(0, 2), vec![0, 1, 2]);
}

#[test]
fn layout_view_tracks_swaps() {
    let mut layout = Layout::from_logical_to_physical(vec![2, 0], 4);
    let before = LayoutView::new(&layout);
    assert_eq!(before.logical_to_physical(), &[2, 0]);
    assert_eq!(
        before.physical_to_logical(),
        &[Some(1), None, Some(0), None]
    );

    layout.swap_physical(2, 3);
    let after = LayoutView::new(&layout);
    assert_eq!(after.physical_of_logical(0), 3);
    assert_eq!(after.logical_of_physical(2), None);
    assert_eq!(after.logical_of_physical(3), Some(0));
}

#[test]
fn rejects_invalid_initial_layout() {
    let target = RoutingTarget::new(2, &[[0, 1]]).unwrap();
    let qasm = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[2] q;\ncx q[0], q[1];\n";
    let mut policy = InvalidInitialLayoutPolicy;
    let err = route_qasm3_with_policy(qasm, &target, &mut policy, RouteOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("multiple logical qubits"));
}

#[test]
fn invalid_swap_is_rejected() {
    let target = RoutingTarget::new(3, &[[0, 1], [1, 2]]).unwrap();
    let qasm = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\ncx q[0], q[2];\n";
    let mut policy = InvalidSwapPolicy;
    let err = route_qasm3_with_policy(qasm, &target, &mut policy, RouteOptions::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("invalid swap"));
}

#[test]
fn none_from_choose_best_swap_uses_fallback_progress() {
    let target = RoutingTarget::new(3, &[[0, 1], [1, 2]]).unwrap();
    let qasm = "OPENQASM 3.0;\ninclude \"stdgates.inc\";\nqubit[3] q;\ncx q[0], q[2];\n";
    let mut policy = FallbackOnlyPolicy;
    let routed = route_qasm3_with_policy(
        qasm,
        &target,
        &mut policy,
        RouteOptions {
            layout_trials: 1,
            routing_trials: 1,
            ..RouteOptions::default()
        },
    )
    .unwrap();
    assert!(routed.swap_count >= 1);
    assert_eq!(routed.initial_mapping.len(), 3);
    assert!(routed.output_circuit.contains("swap") || routed.output_circuit.contains("SWAP"));
}

#[test]
fn disconnected_topology_routes_with_baseline_policy() {
    let target = RoutingTarget::new(4, &[[0, 1], [2, 3]]).unwrap();
    let qasm = concat!(
        "OPENQASM 3.0;\n",
        "include \"stdgates.inc\";\n",
        "qubit[3] q;\n",
        "h q[2];\n",
        "cx q[0], q[1];\n"
    );
    let mut policy = CandidatePolicy::default();
    let routed =
        route_qasm3_with_policy(qasm, &target, &mut policy, RouteOptions::default()).unwrap();
    assert_eq!(routed.swap_count, 0);
    assert_eq!(routed.initial_mapping.len(), 3);
    assert!(routed.output_circuit.contains("cx") || routed.output_circuit.contains("CX"));
}

#[test]
fn disconnected_topology_routes_with_multiple_layout_trials() {
    let target = RoutingTarget::new(4, &[[0, 1], [2, 3]]).unwrap();
    let qasm = concat!(
        "OPENQASM 3.0;\n",
        "include \"stdgates.inc\";\n",
        "qubit[3] q;\n",
        "h q[2];\n",
        "cx q[0], q[1];\n"
    );
    let mut policy = CandidatePolicy::default();
    let routed = route_qasm3_with_policy(
        qasm,
        &target,
        &mut policy,
        RouteOptions {
            layout_trials: 4,
            routing_trials: 1,
            ..RouteOptions::default()
        },
    )
    .unwrap();
    assert_eq!(routed.swap_count, 0);
    assert_eq!(routed.initial_mapping.len(), 3);
    assert!(routed.output_circuit.contains("cx") || routed.output_circuit.contains("CX"));
}
