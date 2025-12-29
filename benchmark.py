import numpy as np
import time
import os
from grainvdb.engine import GrainVDB

def run_standard_benchmark():
    N = 1000000 # 1 Million Vectors
    DIM = 128
    K = 10
    
    print(f"--- GrainVDB Technical Benchmark ---")
    print(f"Dataset: {N:,} vectors | Dimension: {DIM} | Depth: k={K}")
    
    # 1. Setup Data
    print("Generating random float32 vectors...")
    db_vectors = np.random.randn(N, DIM).astype(np.float32)
    query_vec = np.random.randn(DIM).astype(np.float32)
    
    # Pre-normalize for fair CPU comparison
    db_norm = db_vectors / (np.linalg.norm(db_vectors, axis=1, keepdims=True) + 1e-9)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    # 2. CPU Reference (NumPy Optimized)
    print("\n[CPU Reference] Using np.dot + np.argpartition...")
    start_cpu = time.perf_counter()
    # Brute force dot product
    sims = np.dot(db_norm, q_norm)
    # Optimized partial sort (O(N))
    top_k_indices = np.argpartition(sims, -K)[-K:]
    # Final sort of the top-k results
    top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]
    end_cpu = time.perf_counter()
    cpu_wall_ms = (end_cpu - start_cpu) * 1000
    print(f"CPU Wall-Time: {cpu_wall_ms:.2f} ms")

    # 3. GrainVDB Native (Metal + C++)
    try:
        engine = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"GrainVDB Initialization Failed: {e}")
        return

    print("\n[GrainVDB Native] Loading vectors into Unified Memory...")
    engine.add_vectors(db_vectors)
    
    # Warmup
    engine.query(query_vec, k=K)
    
    print("Running measurement loops (n=10)...")
    native_latencies = []
    python_wall_latencies = []
    
    for _ in range(10):
        t_start = time.perf_counter()
        indices, scores, internal_ms = engine.query(query_vec, k=K)
        t_end = time.perf_counter()
        
        native_latencies.append(internal_ms)
        python_wall_latencies.append((t_end - t_start) * 1000)

    avg_internal_ms = np.mean(native_latencies)
    avg_wall_ms = np.mean(python_wall_latencies)
    
    print(f"Internal C++ Benchmark (Metal Loop + Selection): {avg_internal_ms:.2f} ms")
    print(f"Python Wall-Time (End-to-End Bridge): {avg_wall_ms:.2f} ms")
    print(f"Throughput: {1000 / avg_wall_ms:.1f} queries/sec")

    # 4. Correctness Check
    print("\n[Correctness] Verifying top result vs NumPy...")
    if indices[0] == top_k_indices[0]:
        print("✅ MATCH: Native and CPU references agree on Top-1.")
    else:
        print(f"❌ MISMATCH: Native Top-1 is {indices[0]}, CPU Top-1 is {top_k_indices[0]}")
        print(f"   Scores: Native={scores[0]:.4f}, CPU={sims[top_k_indices[0]]:.4f}")

    # 5. Audit Check
    connectivity = engine.audit(indices)
    print(f"\n[Audit] Neighborhood Connectivity: {connectivity:.4f}")
    
    print(f"\n✨ Performance Gain: {cpu_wall_ms / avg_wall_ms:.1f}x (E2E Bridge)")

if __name__ == "__main__":
    run_standard_benchmark()
