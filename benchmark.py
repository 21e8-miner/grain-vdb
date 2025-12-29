import numpy as np
import time
import os
import sys
import platform
import subprocess
from grainvdb import GrainVDB
import pprint

def git_short():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def make_clusters(n: int, d: int, n_clusters: int, sigma: float, seed: int):
    rng = np.random.default_rng(seed)
    # Generate centers
    centers = rng.standard_normal((n_clusters, d), dtype=np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12

    # Assign labels and generate data
    labels = rng.integers(0, n_clusters, size=n, dtype=np.int32)
    x = centers[labels] + sigma * rng.standard_normal((n, d), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x, labels, centers

def topk_cpu(db_norm: np.ndarray, q_norm: np.ndarray, k: int, out_sims: np.ndarray = None):
    # Pre-allocate output buffer if provided to avoid malloc overhead in timing
    if out_sims is None:
        sims = db_norm @ q_norm
    else:
        np.dot(db_norm, q_norm, out=out_sims)
        sims = out_sims
        
    idx = np.argpartition(sims, -k)[-k:]
    idx = idx[np.argsort(sims[idx])[::-1]]
    return idx, sims[idx]

def main():
    N = 1_000_000
    D = 128
    K = 10
    NQ = 100
    WARMUP = 10
    SEED = 42
    N_CLUSTERS = 20
    SIGMA = 0.15

    print("--- GrainVDB Benchmark ---")
    print(f"Commit: {git_short()}")
    print(f"OS/Arch: {platform.system()} {platform.release()} / {platform.machine()}")
    print(f"Python: {sys.version.split()[0]} | NumPy: {np.__version__}")
    
    # Thread environment transparency
    print("Thread Env:", {k: os.environ.get(k, "Not Set") for k in ["VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"]})
    
    # BLAS Backend Best-Effort Check
    try:
        blas_info = np.show_config(mode="dicts")
        print("BLAS Backend:", blas_info.get("Build Dependencies", {}).get("blas", {}).get("name", "Unknown"))
    except:
        pass

    print(f"Params: N={N:,} | D={D} | K={K} | Queries={NQ} | Clusters={N_CLUSTERS}")

    # 1. Data Generation (Clustered & Pre-Normalized)
    print("\n[1] Data Generation (Clustered)...")
    db_norm, _, centers = make_clusters(N, D, N_CLUSTERS, SIGMA, SEED)
    
    # Generate queries from cluster centers with some jitter
    rng = np.random.default_rng(SEED + 1)
    q_labels = rng.integers(0, N_CLUSTERS, size=NQ, dtype=np.int32)
    q_norm = centers[q_labels] + SIGMA * rng.standard_normal((NQ, D), dtype=np.float32)
    q_norm /= np.linalg.norm(q_norm, axis=1, keepdims=True) + 1e-12

    # 2. CPU Baseline (Wall-Time per Query)
    print("\n[2] CPU Baseline (np.dot + np.argpartition + prealloc)")
    # Warmup
    sims_buffer = np.empty(N, dtype=np.float32)
    for i in range(WARMUP):
        topk_cpu(db_norm, q_norm[i % NQ], K, out_sims=sims_buffer)

    cpu_ms = []
    cpu_topk = []
    
    for i in range(NQ):
        t0 = time.perf_counter()
        idx, _ = topk_cpu(db_norm, q_norm[i], K, out_sims=sims_buffer)
        dt = (time.perf_counter() - t0) * 1000.0
        cpu_ms.append(dt)
        cpu_topk.append(idx)

    print(f"    Latency: {np.percentile(cpu_ms, 50):.2f} ms (p50) | {np.percentile(cpu_ms, 95):.2f} ms (p95)")

    # 3. GrainVDB Native (Wall-Time per Query)
    print("\n[3] GrainVDB Native (End-to-End Wall-Time)")
    try:
        vdb = GrainVDB(dim=D)
    except Exception as e:
        print(f"    FATAL: {e}")
        return

    # Use same normalized data and SKIP internal normalization for fairness
    vdb.add_vectors(db_norm, assume_normalized=True)  

    # Warmup
    for i in range(WARMUP):
        vdb.query(q_norm[i % NQ], k=K)

    vdb_ms = []
    vdb_topk = []
    
    for i in range(NQ):
        t0 = time.perf_counter()
        # Query returns (indices, scores, kernel_ms)
        idx, _, _ = vdb.query(q_norm[i], k=K)
        dt = (time.perf_counter() - t0) * 1000.0
        
        vdb_ms.append(dt)
        vdb_topk.append(np.asarray(idx, dtype=np.int64))

    print(f"    Latency: {np.percentile(vdb_ms, 50):.2f} ms (p50) | {np.percentile(vdb_ms, 95):.2f} ms (p95)")

    # 4. Correctness (Recall@K over Queries)
    print("\n[4] Correctness (Recall@K over 100 Queries)")
    recalls = []
    for i in range(NQ):
        a = set(map(int, cpu_topk[i]))
        b = set(map(int, vdb_topk[i]))
        recalls.append(len(a & b) / K)

    mean_recall = np.mean(recalls)
    min_recall = np.min(recalls)
    p50_recall = np.percentile(recalls, 50)
    print(f"    Recall: Mean={mean_recall:.3f} | p50={p50_recall:.3f} | Min={min_recall:.3f}")
    
    if mean_recall < 0.9:
        print("    WARNING: Low recall integration.")
    else:
        print("    PASS: Verified high overlap.")

    # 5. Audit Check (Sanity check on clustered data)
    print("\n[5] Audit Check (Semantic Density)")
    densities = []
    for i in range(NQ): # Check all
        d = vdb.audit(vdb_topk[i])
        densities.append(d)
    
    print(f"    Density Stats: Mean={np.mean(densities):.2f} | p50={np.percentile(densities, 50):.2f} | Max={np.max(densities):.2f}")
    print("    (Expect >0.0 values due to clustered data)")

    speedup = np.percentile(cpu_ms, 50) / np.percentile(vdb_ms, 50)
    print(f"\n✨ Speedup: {speedup:.1f}x (p50)")

if __name__ == "__main__":
    main()
