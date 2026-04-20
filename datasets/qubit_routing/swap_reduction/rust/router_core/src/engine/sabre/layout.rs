#[derive(Debug, Clone)]
pub struct Layout {
    logical_to_physical: Vec<usize>,
    physical_to_logical: Vec<Option<usize>>,
}

impl Layout {
    pub fn from_logical_to_physical(
        logical_to_physical: Vec<usize>,
        num_physical_qubits: usize,
    ) -> Self {
        let mut physical_to_logical = vec![None; num_physical_qubits];
        for (logical, &physical) in logical_to_physical.iter().enumerate() {
            physical_to_logical[physical] = Some(logical);
        }
        Self {
            logical_to_physical,
            physical_to_logical,
        }
    }

    pub fn physical_of_logical(&self, logical: usize) -> usize {
        self.logical_to_physical[logical]
    }

    pub fn logical_of_physical(&self, physical: usize) -> Option<usize> {
        self.physical_to_logical[physical]
    }

    pub fn num_physical(&self) -> usize {
        self.physical_to_logical.len()
    }

    pub fn logical_to_physical_map(&self) -> &[usize] {
        &self.logical_to_physical
    }

    pub fn physical_to_logical_map(&self) -> &[Option<usize>] {
        &self.physical_to_logical
    }

    pub fn swap_physical(&mut self, a: usize, b: usize) {
        self.physical_to_logical.swap(a, b);
        if let Some(logical) = self.physical_to_logical[a] {
            self.logical_to_physical[logical] = a;
        }
        if let Some(logical) = self.physical_to_logical[b] {
            self.logical_to_physical[logical] = b;
        }
    }
}
