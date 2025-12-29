# GrainVDB 🌾
### Native Metal Manifold Engine for Apple Silicon
**Hardware-Accelerated Similarity Search & Neighborhood Consistency Audits**

GrainVDB is an industrial-grade local engine for vector search, written in **Native Objective-C++ and Metal**. It is designed to exploit the **Unified Memory Architecture** of Apple Silicon for zero-copy similarity resolution and high-throughput manifold processing.

---

## 📊 Benchmarks (1 Million x 128D Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| Query Latency (k=10) | ~21 ms | **~3.3 ms** |
| Throughput | 47.6 req/s | **301.2 req/s** |

**Hardware**: MacBook M2 (Unified Memory).
**Methodology**: Measurements denote the cost of similarity computation and top-k selection on pre-normalized unit vectors. 
- **CPU Baseline**: Uses `np.argpartition` for efficient partial sort (O(N) complexity).
- **GrainVDB**: Direct Metal dispatch with shared memory buffers and GPU-side priority queue selection.

---

## 🔬 Core Technologies

### 1. Unified Memory Optimization
Unlike traditional databases that move data over PCIe, GrainVDB maps its vector buffers directly into the GPU's address space. This "Native Bridge" eliminates serialization overhead and allows for sub-4ms resolution on million-vector manifolds.

### 2. Neighborhood Consistency Audit (.audit())
Standard k-NN retrieval can lead to "semantic noise" where similar results are pulled from logically inconsistent contexts. GrainVDB implements a **Topological Consistency Audit**:
- It constructs a local similarity matrix of the retrieved results.
- It calculates the **Gluing Energy** (Algebraic Connectivity).
- A low score signals a "Context Fracture," providing a data-driven filter for RAG halluncination mitigation.

---

## 🚀 Quick Start

```python
from grain_vdb import GrainVDB
import numpy as np

# Initialize Native Core
vdb = GrainVDB(dim=128)

# Ingest data (Normalized internally)
vectors = np.random.randn(1000, 128)
vdb.add_vectors(vectors)

# Query with sub-4ms latency
scores, indices, latency = vdb.query(np.random.randn(128), k=10)

# Audit for context consistency
connectivity = vdb.audit(indices)
```

---

## 🏗️ Technical Roadmap
- [ ] Quasicrystal Phase Coding: High-dimensional quantization for 16x compression.
- [ ] Sheaf-theoretic Complex: Formal Çech cohomology for multi-hop RAG verification.

---

**Author**: Adam Sussman  
**License**: Proprietary / Early Access
