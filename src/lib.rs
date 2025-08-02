use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod distance;
mod vector;
mod index;
mod ivf;

use crate::ivf::IVFIndex;
use crate::index::VectorIndex;
use crate::vector::Vector;

#[pyclass]
struct PyVectorIndex {
    index: VectorIndex,
}

#[pyfunction]
fn load_vectors(path: String) -> PyResult<Vec<(usize, Vec<f32>)>> {
    let vectors = crate::ivf::load_vectors_from_csv(&path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("CSV Load Error: {}", e))
    })?;
    Ok(vectors.into_iter().map(|v| (v.id, v.data)).collect())
}

#[pymethods]
impl PyVectorIndex {
    #[new]
    fn new() -> Self {
        Self {
            index: VectorIndex::new(),
        }
    }

    fn insert(&mut self, id: usize, data: Vec<f32>) {
        self.index.insert(Vector::new(id, data));
    }

    fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(usize, f32)> {
        self.index.search(&query, top_k)
    }
}

#[pyclass]
struct PyIVFIndex {
    inner: IVFIndex,
}

#[pymethods]
impl PyIVFIndex {
    #[new]
    fn new(k: usize) -> Self {
        Self {
            inner: IVFIndex::new(k),
        }
    }

    fn train(&mut self, data: Vec<(usize, Vec<f32>)>) {
        let vectors: Vec<Vector> = data.into_iter()
            .map(|(id, v)| Vector { id, data: v })
            .collect();
        self.inner.train(&vectors);
    }
    #[pyo3(name="search_parallel")]
    fn search_parallel(&self, query: Vec<f32>, top_k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        self.inner.search_parallel(&query, top_k, nprobe)
    }
    #[pyo3(name="search_vanilla")]
    fn search_vanilla(&self, query: Vec<f32>, top_k: usize, nprobe: usize) -> Vec<(usize, f32)> {
        self.inner.search_vanilla(&query, top_k, nprobe)
    }
}

#[pymodule]
fn motheroffline(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyVectorIndex>()?;
    m.add_class::<PyIVFIndex>()?;
    m.add_function(wrap_pyfunction!(load_vectors, m)?)?;
    Ok(())
}
