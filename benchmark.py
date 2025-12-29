import numpy as np
import time
from grainvdb import GrainVDB

def run_technical_audit():
    N = 1000000 # 1 Million Vectors
    DIM = 128
    K = 10
    TRIALS = 10
    
    print(f"--- GrainVDB Engineering Benchmark ---")
    print(f"Dataset Scale: {N:,} vectors | Width: {DIM} dimensions | Depth: k={K}")
    
    # 1. Synthesize Data
    print("Generating random float32 vectors...")
    db_raw = np.random.randn(N, DIM).astype(np.float32)
    q_raw = np.random.randn(DIM).astype(np.float32)
    
    # 2. Optimized CPU Baseline
    # Pre-normalize DB and Query once to isolate strictly search/selection time.
    db_norm = db_raw / (np.linalg.norm(db_raw, axis=1, keepdims=True) + 1e-9)
    q_norm = q_raw / (np.linalg.norm(q_raw) + 1e-9)

    print("\n[CPU Reference] Executing np.dot + np.argpartition (O(N) selection)...")
    cpu_times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        sims = np.dot(db_norm, q_norm)
        top_k_idx = np.argpartition(sims, -K)[-K:]
        # Sorting the final k results for a standard ranking view
        top_k_idx = top_k_idx[np.argsort(sims[top_k_idx])[::-1]]
        cpu_times.append((time.perf_counter() - t0) * 1000)
    
    cpu_median = np.median(cpu_times)
    print(f"CPU Median Wall-Time: {cpu_median:.2f} ms")

    # 3. Native Core (GrainVDB)
    try:
        vdb = GrainVDB(dim=DIM)
    except Exception as e:
        print(f"Engine Load Failed: {e}")
        return

    print("\n[GrainVDB Native] Loading data into Unified Memory...")
    vdb.add_vectors(db_raw)
    
    # Warmup
    vdb.query(q_raw, k=K)
    
    print(f"Executing native measurement loops (n={TRIALS})...")
    vdb_times = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        indices, scores, _ = vdb.query(q_raw, k=K)
        vdb_times.append((time.perf_counter() - t0) * 1000)
        
    vdb_median = np.median(vdb_times)
    print(f"Native Median Wall-Time (E2E Bridge): {vdb_median:.2f} ms")

    # 4. Correctness Audit
    print("\n[Audit] Integrity & Correctness Check...")
    # Tolerance due to FP16 vs FP32 accumulation
    diff = abs(scores[0] - sims[top_k_idx[0]])
    if indices[0] == top_k_idx[0]:
        print(f"✅ EXACT MATCH: Native and CPU Top-1 index match ({indices[0]}).")
    elif diff < 1e-3:
         print(f"✅ SOFT MATCH: Indices differ but score delta is negligible ({diff:.6f}).")
    else:
        print(f"❌ MISMATCH: Score delta = {diff:.6f}. Check precision flow.")

    # 5. Connectivity Audit
    consensus = vdb.audit(indices)
    print(f"[Audit] Semantic Consensus (Neighborhood Connectivity): {consensus:.4f}")
    
    print(f"\n✨ Performance Gain: {cpu_median / vdb_median:.1f}x vs optimized CPU Reference.")

if __name__ == "__main__":
    run_technical_audit()
