import numpy as np
import time
import sys
from grainvdb import GrainVDB

def run_performance_audit():
    N = 1000000  # 1 Million Vectors
    DIM = 128
    K = 10
    
    print(f"--- GrainVDB Technical Audit ---")
    print(f"Manifold Depth: {N:,} vectors | Width: {DIM} dimensions")
    
    # 1. Dataset Generation
    print("Generating random float32 vectors...")
    db_raw = np.random.randn(N, DIM).astype(np.float32)
    query_raw = np.random.randn(DIM).astype(np.float32)
    
    # 2. Optimized CPU Baseline (NumPy / OpenBLAS)
    # We pre-normalize outside the timed region to isolate the SEARCH performance.
    print("[CPU Baseline] Pre-normalizing vectors...")
    db_norm = db_raw / (np.linalg.norm(db_raw, axis=1, keepdims=True) + 1e-9)
    query_norm = query_raw / (np.linalg.norm(query_raw) + 1e-9)

    print("[CPU Baseline] Executing np.dot + np.argpartition...")
    t_start = time.perf_counter()
    
    # Brute-force cosine similarity (N x D)
    sims = np.dot(db_norm, query_norm)
    
    # O(N) Selection of top K elements
    top_k_idx = np.argpartition(sims, -K)[-K:]
    
    # Final sort of top K (negligible)
    top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
    
    t_end = time.perf_counter()
    cpu_ms = (t_end - t_start) * 1000
    print(f"CPU Baseline Wall-Time: {cpu_ms:.2f} ms")

    # 3. Native Core Bridge (GrainVDB)
    try:
        vdb = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"Engine Initialization Failed: {e}")
        return

    print("\n[Native Core] Ingesting vectors into Unified Memory Buffer...")
    vdb.add_vectors(db_raw)
    
    # Warmup loop
    vdb.query(query_raw, k=K)
    
    print("Running timed measurements (10 iterations)...")
    wall_latencies = []
    
    for _ in range(10):
        t1 = time.perf_counter()
        indices, scores, _ = vdb.query(query_raw, k=K)
        t2 = time.perf_counter()
        wall_latencies.append((t2 - t1) * 1000)

    avg_wall_ms = np.mean(wall_latencies)
    print(f"GrainVDB Wall-Time (inc. Python/C++ Bridge + Top-k): {avg_wall_ms:.2f} ms")

    # 4. Correctness Check
    print("\n[Audit] Integrity Verification...")
    # NOTE: FP16 (Metal) vs FP32 (NumPy) can lead to slight score variations or tie reordering.
    # We verify that the Top-1 results are within a reasonable tolerance or same index.
    score_diff = abs(scores[0] - sims[top_k_idx[0]])
    
    if indices[0] == top_k_idx[0]:
        print(f"✅ EXACT MATCH: Native and CPU Top-1 index match ({indices[0]}).")
    elif score_diff < 1e-4:
        print(f"✅ NEAR MATCH: Scores are equivalent within 1e-4 ({scores[0]:.4f} vs {sims[top_k_idx[0]]:.4f}).")
    else:
        print(f"❌ MISMATCH: Check implementation. Diff={score_diff:.6f}")

    # 5. Audit Logic Check
    connectivity = vdb.audit(indices)
    print(f"[Audit] Semantic Consensus (Neighborhood Audit): {connectivity:.4f}")
    
    print(f"\n✨ Performance Gain: {cpu_ms / avg_wall_ms:.1f}x vs optimized CPU Reference.")

if __name__ == "__main__":
    run_performance_audit()
