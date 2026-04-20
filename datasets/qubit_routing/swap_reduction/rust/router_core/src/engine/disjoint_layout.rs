use std::collections::VecDeque;

use super::routing_target::RoutingTarget;

pub fn connected_components(target: &RoutingTarget) -> Vec<Vec<usize>> {
    let mut seen = vec![false; target.num_qubits];
    let mut components = Vec::new();

    for start in 0..target.num_qubits {
        if seen[start] {
            continue;
        }
        let mut queue = VecDeque::new();
        let mut component = Vec::new();
        seen[start] = true;
        queue.push_back(start);

        while let Some(cur) = queue.pop_front() {
            component.push(cur);
            for &next in target.neighbors(cur) {
                if seen[next] {
                    continue;
                }
                seen[next] = true;
                queue.push_back(next);
            }
        }

        component.sort_unstable();
        components.push(component);
    }

    components.sort_by_key(|c| std::cmp::Reverse(c.len()));
    components
}
