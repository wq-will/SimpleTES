use ahash::RandomState;
use indexmap::IndexMap;

/// A container for the current non-routable parts of the front layer.
#[derive(Debug, Clone)]
pub struct FrontLayer {
    nodes: IndexMap<usize, [usize; 2], RandomState>,
    qubits: Vec<Option<(usize, usize)>>,
}

impl FrontLayer {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            nodes: IndexMap::with_capacity_and_hasher(num_qubits / 2, RandomState::default()),
            qubits: vec![None; num_qubits],
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn qubits(&self) -> &[Option<(usize, usize)>] {
        &self.qubits
    }

    pub fn insert(&mut self, index: usize, qubits: [usize; 2]) {
        let [a, b] = qubits;
        self.qubits[a] = Some((index, b));
        self.qubits[b] = Some((index, a));
        self.nodes.insert(index, qubits);
    }

    pub fn remove(&mut self, index: &usize) {
        let Some([a, b]) = self.nodes.swap_remove(index) else {
            return;
        };
        self.qubits[a] = None;
        self.qubits[b] = None;
    }

    pub fn is_active(&self, qubit: usize) -> bool {
        self.qubits[qubit].is_some()
    }

    pub fn score_delta(&self, swap: (usize, usize), dist: &[Vec<u32>]) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        if let Some((_, c)) = self.qubits[a] {
            total += (dist[b][c] as f64) - (dist[a][c] as f64);
        }
        if let Some((_, c)) = self.qubits[b] {
            total += (dist[a][c] as f64) - (dist[b][c] as f64);
        }
        total
    }

    pub fn total_score(&self, dist: &[Vec<u32>]) -> f64 {
        self.iter()
            .map(|(_, &[a, b])| f64::from(dist[a][b]))
            .sum::<f64>()
    }

    pub fn apply_swap(&mut self, swap: (usize, usize)) {
        let (a, b) = swap;
        match (self.qubits[a], self.qubits[b]) {
            (Some((index1, _)), Some((index2, _))) if index1 == index2 => {
                if let Some(entry) = self.nodes.get_mut(&index1) {
                    *entry = [entry[1], entry[0]];
                }
                return;
            }
            _ => {}
        }
        if let Some((index, c)) = self.qubits[a] {
            self.qubits[c] = Some((index, b));
            if let Some(entry) = self.nodes.get_mut(&index) {
                *entry = if *entry == [a, c] { [b, c] } else { [c, b] };
            }
        }
        if let Some((index, c)) = self.qubits[b] {
            self.qubits[c] = Some((index, a));
            if let Some(entry) = self.nodes.get_mut(&index) {
                *entry = if *entry == [b, c] { [a, c] } else { [c, a] };
            }
        }
        self.qubits.swap(a, b);
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&usize, &[usize; 2])> {
        self.nodes.iter()
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = &usize> {
        self.nodes.keys()
    }

    pub fn iter_active(&self) -> impl Iterator<Item = &usize> {
        self.nodes.values().flatten()
    }
}

#[derive(Debug, Clone)]
pub struct ExtendedSet {
    qubits: Vec<Vec<usize>>,
    len: usize,
}

impl ExtendedSet {
    pub fn new(num_qubits: usize) -> Self {
        Self {
            qubits: vec![Vec::new(); num_qubits],
            len: 0,
        }
    }

    pub fn push(&mut self, qubits: [usize; 2]) {
        let [a, b] = qubits;
        self.qubits[a].push(b);
        self.qubits[b].push(a);
        self.len += 1;
    }

    pub fn score_delta(&self, swap: (usize, usize), dist: &[Vec<u32>]) -> f64 {
        let (a, b) = swap;
        let mut total = 0.0;
        for other in &self.qubits[a] {
            if *other == b {
                continue;
            }
            total += (dist[b][*other] as f64) - (dist[a][*other] as f64);
        }
        for other in &self.qubits[b] {
            if *other == a {
                continue;
            }
            total += (dist[a][*other] as f64) - (dist[b][*other] as f64);
        }
        total
    }

    pub fn total_score(&self, dist: &[Vec<u32>]) -> f64 {
        self.qubits
            .iter()
            .enumerate()
            .flat_map(|(a, others)| others.iter().map(move |b| f64::from(dist[a][*b])))
            .sum::<f64>()
            * 0.5
    }

    pub fn clear(&mut self) {
        for others in &mut self.qubits {
            others.clear();
        }
        self.len = 0;
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn apply_swap(&mut self, swap: (usize, usize)) {
        let (a, b) = swap;
        for other in &mut self.qubits[a] {
            if *other == b {
                *other = a;
            }
        }
        for other in &mut self.qubits[b] {
            if *other == a {
                *other = b;
            }
        }
        self.qubits.swap(a, b);
    }
}
