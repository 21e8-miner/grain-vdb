import numpy as np
import time
from grainvdb.core.gv1_bridge import GV1Engine

# GrainVDB: Native Metal-Accelerated Manifold Engine
# ================================================

class GrainVDB:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.engine = GV1Engine(rank=dim)
        print(f"GrainVDB: Initialized Native Core (Metal/Unified Memory)")

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the native device buffer."""
        # Ensure vectors are normalized for cosine similarity in this alpha version
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / (norms + 1e-9)
        self.engine.feed(vectors)

    def query(self, query_vec: np.ndarray, k: int = 5):
        """Query using Native Metal kernel."""
        # Normalize probe
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        
        indices, scores, latency = self.engine.fold(query_vec, top=k)
        return scores, indices, latency

    def audit(self, indices: np.ndarray):
        """Perform a neighborhood consistency audit."""
        # The bridge handles the pointer mapping to C++
        # uint64_t* map, uint32_t count
        ptr = indices.astype(np.uint64).ctypes.data_as(self.engine.lib.gv1_topology_audit.argtypes[1])
        return self.engine.lib.gv1_topology_audit(self.engine.ctx, ptr, len(indices))

def run_benchmark():
    N = 1000000 # 1 Million Vectors
    DIM = 128
    K = 10
    
    print(f"Generating {N:,} random vectors (DIM={DIM})...")
    db_vectors = np.random.randn(N, DIM).astype(np.float32)
    query_vec = np.random.randn(DIM).astype(np.float32)
    
    # 1. CPU Baseline (NumPy / Optimized Partition)
    # We pre-normalize to measure strictly the similarity + top-k extraction
    db_cpu = db_vectors / (np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9)
    q_cpu = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    print("\nRunning CPU Benchmark (NumPy Partition)...")
    start = time.perf_counter()
    sims = np.dot(db_cpu, q_cpu)
    top_k_idx = np.argpartition(sims, -K)[-K:]
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    cpu_time = time.perf_counter() - start
    print(f"CPU Time: {cpu_time*1000:.2f} ms")

    # 2. GrainVDB (Native Metal)
    vdb = GrainVDB(dim=DIM)
    print("\nLoading vectors into Native Metal Buffer...")
    vdb.add_vectors(db_vectors)
    
    # Warmup
    vdb.query(query_vec, k=K)
    
    # Benchmark
    loops = 10
    total_time = 0
    for i in range(loops):
        _, _, elapsed = vdb.query(query_vec, k=K)
        total_time += elapsed
    
    avg_metal_time = total_time / loops
    print(f"GrainVDB (Native Metal) Time: {avg_metal_time:.2f} ms")
    print(f"\nSPEEDUP: {cpu_time*1000 / avg_metal_time:.1f}x")

    # 3. Audit Verification
    _, indices, _ = vdb.query(query_vec, k=K)
    consistency = vdb.audit(indices)
    print(f"\nNeighborhood Consistency (Audit): {consistency:.4f}")

if __name__ == "__main__":
    run_benchmark()
