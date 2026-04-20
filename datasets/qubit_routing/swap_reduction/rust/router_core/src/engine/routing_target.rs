use std::collections::{HashSet, VecDeque};

use crate::engine::RouterError;
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};

#[derive(Debug, Clone)]
pub struct RoutingTarget {
    pub num_qubits: usize,
    neighbors: Vec<Vec<usize>>,
    dist: Vec<Vec<u32>>,
    edges: Vec<[usize; 2]>,
    degrees: Vec<usize>,
    components: Vec<Vec<usize>>,
    component_of_physical: Vec<usize>,
    edge_set: HashSet<(usize, usize)>,
    graph: UnGraph<(), ()>,
}

impl RoutingTarget {
    pub fn new(num_qubits: usize, edges: &[[usize; 2]]) -> Result<Self, RouterError> {
        if num_qubits == 0 {
            return Err(RouterError::Topology(
                "num_qubits must be greater than zero".to_string(),
            ));
        }

        let mut neighbors = vec![Vec::<usize>::new(); num_qubits];
        let mut edge_set = HashSet::new();
        let mut canonical_edges = Vec::with_capacity(edges.len());
        let mut graph = UnGraph::<(), ()>::with_capacity(num_qubits, edges.len());
        for _ in 0..num_qubits {
            graph.add_node(());
        }

        for edge in edges {
            let (a, b) = (edge[0], edge[1]);
            if a >= num_qubits || b >= num_qubits {
                return Err(RouterError::Topology(format!(
                    "edge [{a}, {b}] is out of range for {num_qubits} qubits"
                )));
            }
            if a == b {
                return Err(RouterError::Topology(format!(
                    "self-loop edge [{a}, {b}] is not allowed"
                )));
            }
            let sorted = if a < b { (a, b) } else { (b, a) };
            if edge_set.insert(sorted) {
                neighbors[sorted.0].push(sorted.1);
                neighbors[sorted.1].push(sorted.0);
                canonical_edges.push([sorted.0, sorted.1]);
                graph.add_edge(NodeIndex::new(sorted.0), NodeIndex::new(sorted.1), ());
            }
        }

        for n in &mut neighbors {
            n.sort_unstable();
        }
        canonical_edges.sort_unstable();

        let dist = Self::build_all_pairs_distances(num_qubits, &neighbors);
        let degrees = neighbors.iter().map(Vec::len).collect::<Vec<_>>();
        let (components, component_of_physical) =
            Self::build_connected_components(num_qubits, &neighbors);

        Ok(Self {
            num_qubits,
            neighbors,
            dist,
            edges: canonical_edges,
            degrees,
            components,
            component_of_physical,
            edge_set,
            graph,
        })
    }

    pub fn is_adjacent(&self, a: usize, b: usize) -> bool {
        let key = if a < b { (a, b) } else { (b, a) };
        self.edge_set.contains(&key)
    }

    pub fn neighbors(&self, q: usize) -> &[usize] {
        &self.neighbors[q]
    }

    pub fn edges(&self) -> &[[usize; 2]] {
        &self.edges
    }

    pub fn degree(&self, qubit: usize) -> usize {
        self.degrees[qubit]
    }

    pub fn max_degree(&self) -> usize {
        self.degrees.iter().copied().max().unwrap_or(0)
    }

    pub fn connected_components(&self) -> &[Vec<usize>] {
        &self.components
    }

    pub fn component_of_physical(&self, physical: usize) -> usize {
        self.component_of_physical[physical]
    }

    pub fn distance(&self, a: usize, b: usize) -> u32 {
        self.dist[a][b]
    }

    pub fn distances(&self) -> &[Vec<u32>] {
        &self.dist
    }

    pub fn graph(&self) -> &UnGraph<(), ()> {
        &self.graph
    }

    pub fn shortest_path(&self, a: usize, b: usize) -> Vec<usize> {
        if a >= self.num_qubits || b >= self.num_qubits {
            return Vec::new();
        }
        if a == b {
            return vec![a];
        }

        let mut parent = vec![usize::MAX; self.num_qubits];
        let mut visited = vec![false; self.num_qubits];
        let mut queue = VecDeque::new();

        visited[a] = true;
        queue.push_back(a);

        while let Some(cur) = queue.pop_front() {
            for &next in &self.neighbors[cur] {
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                parent[next] = cur;
                if next == b {
                    let mut path = vec![b];
                    let mut p = b;
                    while p != a {
                        p = parent[p];
                        path.push(p);
                    }
                    path.reverse();
                    return path;
                }
                queue.push_back(next);
            }
        }

        Vec::new()
    }

    fn build_all_pairs_distances(num_qubits: usize, neighbors: &[Vec<usize>]) -> Vec<Vec<u32>> {
        let mut out = vec![vec![u32::MAX; num_qubits]; num_qubits];
        for start in 0..num_qubits {
            let mut queue = VecDeque::new();
            out[start][start] = 0;
            queue.push_back(start);
            while let Some(cur) = queue.pop_front() {
                let next_dist = out[start][cur].saturating_add(1);
                for &next in &neighbors[cur] {
                    if next_dist < out[start][next] {
                        out[start][next] = next_dist;
                        queue.push_back(next);
                    }
                }
            }
        }
        out
    }

    fn build_connected_components(
        num_qubits: usize,
        neighbors: &[Vec<usize>],
    ) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut seen = vec![false; num_qubits];
        let mut components = Vec::new();

        for start in 0..num_qubits {
            if seen[start] {
                continue;
            }
            let mut queue = VecDeque::new();
            let mut component = Vec::new();
            seen[start] = true;
            queue.push_back(start);

            while let Some(cur) = queue.pop_front() {
                component.push(cur);
                for &next in &neighbors[cur] {
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

        components.sort_by_key(|component| std::cmp::Reverse(component.len()));
        let mut component_of_physical = vec![0usize; num_qubits];
        for (component_idx, component) in components.iter().enumerate() {
            for &physical in component {
                component_of_physical[physical] = component_idx;
            }
        }

        (components, component_of_physical)
    }
}
