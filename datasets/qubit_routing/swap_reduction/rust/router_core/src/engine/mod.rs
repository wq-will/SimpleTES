pub mod dense_layout;
pub mod disjoint_layout;
pub mod routing_target;
pub mod sabre;

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::candidate::Policy;
use routing_target::RoutingTarget;

#[derive(Debug, Clone, Copy)]
pub struct RouteOptions {
    pub seed: u64,
    pub max_swaps_without_progress: usize,
    pub layout_trials: usize,
    pub routing_trials: usize,
}

impl Default for RouteOptions {
    fn default() -> Self {
        Self {
            seed: 7,
            max_swaps_without_progress: 1000,
            layout_trials: 4,
            routing_trials: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RouteOutput {
    pub swap_count: usize,
    pub depth: usize,
    pub twoq_count: usize,
    pub initial_mapping: Vec<usize>,
    pub output_circuit: String,
}

#[derive(Debug, Clone)]
pub enum RouterError {
    Parse(String),
    Topology(String),
    Routing(String),
}

impl Display for RouterError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RouterError::Parse(msg) => write!(f, "qasm3 parse error: {msg}"),
            RouterError::Topology(msg) => write!(f, "invalid topology: {msg}"),
            RouterError::Routing(msg) => write!(f, "routing failed: {msg}"),
        }
    }
}

impl Error for RouterError {}

pub fn route_qasm3_with_policy<P: Policy>(
    qasm3: &str,
    target: &RoutingTarget,
    policy: &mut P,
    options: RouteOptions,
) -> Result<RouteOutput, RouterError> {
    sabre::route::route_qasm3(qasm3, target, policy, options)
}
