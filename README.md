# GrainVDB 🌾
### Native Metal-Accelerated Vector Engine for Apple Silicon
**Verified High-Performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built specifically for macOS and Apple Silicon. It bypasses the overhead of standard frameworks by using a direct Objective-C++/Metal bridge, enabling efficient brute-force similarity search on massive vector manifolds using hardware-accelerated SIMD.

---

## 📊 Performance (1 Million x 128D Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| **End-to-End Latency** | ~21.9 ms | **~5.5 ms** |
| **Throughput** | 45.6 req/s | **181.8 req/s** |

**Hardware**: MacBook M2 (Unified Memory).  
**Methodology**:
- **Wall-Time Measurement**: Latency is measured end-to-end at the Python boundary, including dylib bridge overhead, GPU execution, and host-side Top-k selection.
- **CPU Baseline**: Optimized NumPy implementation using `np.argpartition` ($O(N)$ partial sort) and `np.dot` on pre-normalized vectors.
- **GrainVDB**: Brute-force resolution on GPU (Metal) + Priority Queue selection on CPU. Unified Memory ensures zero-copy access between the device and host buffers.

---

## 🔬 Core Architecture

### 1. Unified Memory Bridge
GrainVDB exploits the shared memory architecture of Apple M-series chips. By mapping host-resident Python buffers directly into the GPU's address space using `storageModeShared` MTLBuffers, the engine eliminates the serialization and PCIe copy bottlenecks typical of discrete GPU systems.

### 2. Custom Metal Kernels
Similarity resolution is performed by vectorized `half4` kernels written in Metal Shading Language (MSL). These kernels are designed to maximize the memory bandwidth of the M-series SOC while performing low-precision (FP16) dot-product accumulation.

### 3. Neighborhood Consistency Audit
GrainVDB includes a built-in topological audit layer that calculates **Neighborhood Connectivity** (density of pairwise relationships among retrieved results). This heuristic identifies "Semantic Fractures" where retrieved context is disjointed, signaling higher hallucination risk in RAG applications.

---

## 🚀 Getting Started

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
from grainvdb import GrainVDB
import numpy as np

# Initialize engine for 128-dimensional vectors
vdb = GrainVDB(dim=128)

# Add 1 million vectors (Normalized internally)
data = np.random.randn(1000000, 128).astype(np.float32)
vdb.add_vectors(data)

# Query in ~5.5ms
indices, scores, e2e_ms = vdb.query(np.random.randn(128), k=10)

# Verify neighborhood consistency
density = vdb.audit(indices)
```

---

## 🏗️ Engineering Roadmap
- [ ] **Quantized Storage (INT8/INT4)**: Support for 10M+ vector manifolds on memory-constrained systems.
- [ ] **GPU-Side Selection**: Implementing Heap-sort/Selection on-GPU to further reduce bridge overhead.
- [ ] **Rust Integration**: High-performance bindings for systems-level orchestration.

---

**Author**: Adam Sussman  
**License**: MIT
