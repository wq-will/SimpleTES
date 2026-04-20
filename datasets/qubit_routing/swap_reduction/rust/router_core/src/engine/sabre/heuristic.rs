#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetScaling {
    Constant,
    Size,
}

#[derive(Debug, Clone, Copy)]
pub struct Heuristic {
    pub basic_weight: f64,
    pub lookahead_weight: f64,
    pub lookahead_size: usize,
    pub set_scaling: SetScaling,
    pub use_decay: bool,
    pub decay_increment: f64,
    pub decay_reset: usize,
    pub best_epsilon: f64,
    pub attempt_limit: usize,
}

impl Default for Heuristic {
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
            attempt_limit: 1000,
        }
    }
}
