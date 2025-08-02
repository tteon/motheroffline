use crate::vector::Vector;
use crate::distance::l2_distance;

pub struct VectorIndex {
    pub vectors: Vec<Vector>,
}

impl VectorIndex {
    pub fn new() -> Self {
        Self { vectors: Vec::new() }
    }

    pub fn insert(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut results: Vec<(usize, f32)> = self.vectors
            .iter()
            .map(|v| (v.id, l2_distance(&v.data, query)))
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results
    }
}
