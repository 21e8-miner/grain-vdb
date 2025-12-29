# GrainVDB 🌾
### Native Metal-Accelerated Vector Engine for Apple Silicon
**High-Performance Similarity Search via Unified Memory Optimization**

GrainVDB is a native vector engine built specifically for macOS and Apple Silicon. It utilizes a direct Objective-C++/Metal bridge to bypass the overhead of standard frameworks, enabling efficient brute-force similarity search on massive vector manifolds using hardware-accelerated SIMD.

---

## 📊 Performance (1 Million x 128D Vectors)
| Metric | CPU (NumPy Partition) | **GrainVDB (Native Metal)** |
|--------|----------------------|-----------------------|
| **Latency (Median)** | ~21.5 ms | **~5.1 ms** |
| **Throughput** | 46.5 req/s | **196.1 req/s** |

**Hardware**: MacBook M2 (Unified Memory).  
**Methodology**:
- **Wall-Time measurement**: Latency is measured at the Python boundary via `time.perf_counter()`, including bridge overhead and Top-k selection.
- **CPU Baseline**: Optimized NumPy implementation using pre-normalized vectors and `np.argpartition` ($O(N)$ partial sort).
- **GrainVDB Native**: Brute-force resolution on GPU (Metal, FP16) + Priority Queue selection on CPU. Unified Memory ensures zero-copy access between device and host buffers.

---

## 🔬 Core Architecture

### 1. Unified Memory Mapping
GrainVDB exploits the shared memory architecture of Apple M-series chips. By mapping host-resident Python/NumPy buffers directly into the GPU's address space using `storageModeShared` MTLBuffers, the engine eliminates the serialization and copy bottlenecks typical of PCIe-based discrete GPU systems.

### 2. Custom Metal Kernels
Similarity resolution is performed by vectorized `half4` kernels written in Metal Shading Language (MSL). These kernels are designed to maximize the memory bandwidth of the M-series SOC while performing low-precision (FP16) dot-product accumulation for 4x instruction throughput compared to standard FP32 paths.

### 3. Neighborhood Consistency Audit
The engine includes a built-in topological audit layer that calculates the **Neighborhood Connectivity** density of retrieved results. This heuristic helps identify "Semantic Fractures"—where top results are disjointed—signaling potential hallucination risk in RAG applications.

---

## 🚀 Getting Started

### 1. Build Native Core
GrainVDB requires a local build to link against your system's Metal frameworks.
```bash
chmod +x build.sh
./build.sh
```

### 2. Run Benchmark
Verify performance and mathematical correctness on your machine.
```bash
python3 benchmark.py
```

### 3. Basic usage
```python
from grainvdb import GrainVDB
import numpy as np

# Initialize for 128-dimensional vectors
vdb = GrainVDB(dim=128)

# Add 1 million vectors (Normalized internally)
data = np.random.randn(1000000, 128).astype(np.float32)
vdb.add_vectors(data)

# High-speed search
indices, scores, latency_ms = vdb.query(np.random.randn(128), k=10)

# Consistency Audit
density = vdb.audit(indices)
```

---

## 🏗️ Technical Roadmap
- [ ] **Quantized INT8/INT4 Selection**: Support for 10M+ manifolds on memory-constrained devices.
- [ ] **GPU-Side Selection**: Implementing Heapsort/Selection on-GPU to further reduce wall-clock latency.
- [ ] **Graph-Based Neighbors**: Integrating HNSW-style graph traversal for sub-millisecond sub-linear search.

---

**Author**: Adam Sussman  
**License**: MIT
