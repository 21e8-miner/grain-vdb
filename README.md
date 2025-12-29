# GrainVDB 🌾
### Native Metal-Accelerated Vector Engine for Apple Silicon
**High-Performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built for macOS and Apple Silicon. It utilizes a direct Objective-C++/Metal bridge to bypass the overhead of standard AI frameworks, enabling massive-scale similarity discovery with minimal latency.

---

## 📊 Performance (1 Million x 128D Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| **End-to-End Latency** | ~18.3 ms | **~12.2 ms** |
| **Throughput** | 54.6 req/s | **82.0 req/s** |

**Hardware**: MacBook M2 (Unified Memory).  
**Methodology**:
- **Wall-Time Measurement**: Latency is measured end-to-end at the Python boundary, including the bridge overhead and Top-k selection.
- **CPU Baseline**: Highly optimized NumPy implementation using `np.argpartition` ($O(N)$ partial sort) on pre-normalized vectors.
- **GrainVDB**: Brute-force discovery on GPU (Metal) + Priority Queue selection on CPU. Unified Memory ensures zero-copy access between the device and host buffers.

---

## 🔬 Core Architecture

### 1. Unified Memory Bridge
GrainVDB exploits the shared memory architecture of M-series chips. By mapping Python buffers into the GPU's address space using `storageModeShared` MTLBuffers, the engine avoids the data serialization and copy bottlenecks typical of PCIe-based systems.

### 2. Custom Metal Kernels
Similarity resolution is performed by vectorized `half4` kernels written in Metal Shading Language (MSL). These kernels are designed to saturate the GPU's memory bandwidth while maintaining high floating-point throughput.

### 3. Neighborhood Consistency Audit
To mitigate retrieval noise and potential RAG hallucinations, GrainVDB includes a built-in **Audit Layer**. It calculates the **Neighborhood Connectivity Score** (the density of semantic relationships among the top results). A low score indicates a "Context Fracture," signaling that the retrieved results are semantically disjointed.

---

## 🚀 Installation & Usage

### 1. Build from Source
GrainVDB compiles its native core locally to match your hardware's Metal framework version.
```bash
chmod +x build.sh
./build.sh
```

### 2. Run the Technical Benchmark
```bash
python3 benchmark.py
```

### 3. Integration Example
```python
from grainvdb.engine import GrainVDB
import numpy as np

# Initialize engine for 128-dimensional vectors
vdb = GrainVDB(dim=128)

# Add 1 million vectors (Normalized internally)
data = np.random.randn(1000000, 128).astype(np.float32)
vdb.add_vectors(data)

# Query in ~12ms
indices, scores, e2e_ms = vdb.query(np.random.randn(128), k=10)

# Verify neighborhood consistency
density = vdb.audit(indices)
```

---

## 🏗️ Engineering Roadmap
- [ ] **Quantized INT8/INT4 Storage**: Support for 10M+ vector manifolds on memory-constrained devices.
- [ ] **GPU-Side Selection**: Moving the Top-k priority queue into the Metal kernel for sub-3ms resolution.
- [ ] **Native C-API / Rust Bindings**: Direct integration for low-latency systems.

---

**Author**: Adam Sussman  
**License**: MIT
