# motheroffline# motheroffline

**Fast vector search in Rust with Python bindings**  
A toy project to implement vector indexing algorithms (Brute Force, IVF) from scratch in Rust, with PyO3 for Python integration.

---

## ✨ Features

- Written in pure Rust for performance and safety
- IVF (Inverted File Index) implementation
- Brute-force fallback for baseline comparison
- Multi-threaded search using [`rayon`](https://docs.rs/rayon/)
- Python bindings via [`PyO3`](https://pyo3.rs/)
- Easy integration from Python (e.g., Jupyter notebooks)
- CSV loading or synthetic data generation supported

---

## 📦 Installation (Python side)

Build and install the Python module:

```bash
# From inside the root folder
maturin develop  # or
maturin build --release && pip install target/wheels/*.whl
