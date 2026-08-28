#!/usr/bin/env python3
"""
GrainVDB v2.0 - Apple Silicon Benchmark Runner
Benchmarks Adaptive Dual-Engine (CPU Accelerate + Metal GPU),
HNSW graph indexing, batch throughput, and zero-copy mmap.
"""

import argparse
import os
import time
import numpy as np
from grainvdb import GrainVDB, SearchMode, EngineType, HNSWConfig


def run_benchmark(n_vectors: int = 20000, dim: int = 128, n_queries: int = 50, k: int = 10):
    print("=" * 68)
    print("  GrainVDB v2.0 - Apple Silicon Adaptive Benchmark")
    print("=" * 68)
    print(f"  Vectors:        {n_vectors:,}")
    print(f"  Dimension:      {dim}")
    print(f"  Test Queries:   {n_queries}")
    print(f"  Top-K:          {k}")
    print("=" * 68)
    print()

    print(f"Generating {n_vectors:,} random {dim}D normalized vectors...")
    np.random.seed(42)
    raw_data = np.random.randn(n_vectors, dim).astype(np.float32)
    vectors = raw_data / (np.linalg.norm(raw_data, axis=1, keepdims=True) + 1e-12)

    raw_queries = np.random.randn(n_queries, dim).astype(np.float32)
    queries = raw_queries / (np.linalg.norm(raw_queries, axis=1, keepdims=True) + 1e-12)

    # 1. Benchmark CPU Accelerate Fast-Path
    print("\n[1/4] Benchmarking Apple Accelerate CPU Fast-Path...")
    vdb_cpu = GrainVDB(dim=dim, mode=SearchMode.EXACT, engine=EngineType.ACCELERATE)
    t0 = time.perf_counter()
    vdb_cpu.add_vectors(vectors)
    ingest_time = time.perf_counter() - t0
    print(f"  ✓ Ingested {n_vectors:,} vectors in {ingest_time*1000:.1f} ms ({n_vectors/ingest_time:,.0f} vec/s)")

    cpu_latencies = []
    ground_truth_indices = []
    for q in queries:
        res = vdb_cpu.search(q, k=k)
        cpu_latencies.append(res.latency_ms)
        ground_truth_indices.append(set(res.indices))

    cpu_p50 = np.percentile(cpu_latencies, 50)
    cpu_p95 = np.percentile(cpu_latencies, 95)
    cpu_p99 = np.percentile(cpu_latencies, 99)
    print(f"  ✓ Single Query Latency: p50: {cpu_p50:.3f} ms ({cpu_p50*1000:.1f} µs) | p95: {cpu_p95:.3f} ms | p99: {cpu_p99:.3f} ms")

    # 2. Benchmark Metal GPU Engine
    print("\n[2/4] Benchmarking Native Metal GPU Engine...")
    vdb_gpu = GrainVDB(dim=dim, mode=SearchMode.EXACT, engine=EngineType.METAL)
    vdb_gpu.add_vectors(vectors)
    vdb_gpu.warmup()

    gpu_latencies = []
    for q in queries:
        res = vdb_gpu.search(q, k=k)
        gpu_latencies.append(res.latency_ms)

    gpu_p50 = np.percentile(gpu_latencies, 50)
    gpu_p95 = np.percentile(gpu_latencies, 95)
    print(f"  ✓ Single Query Latency: p50: {gpu_p50:.3f} ms | p95: {gpu_p95:.3f} ms (Driver sync bound)")

    # Batch throughput measurement on GPU
    peak_qps = 0.0
    for batch_sz in [16, 64, 128]:
        batch_queries = queries[:min(batch_sz, n_queries)]
        t_start = time.perf_counter()
        vdb_gpu.search_batch(batch_queries, k=k)
        t_end = time.perf_counter()
        elapsed_sec = t_end - t_start
        qps = len(batch_queries) / elapsed_sec
        peak_qps = max(peak_qps, qps)
        print(f"  ✓ Batch Size {len(batch_queries):3d} GPU Throughput: {qps:,.1f} QPS ({elapsed_sec*1000:.2f} ms total)")

    # 3. Benchmark HNSW Index
    print("\n[3/4] Benchmarking HNSW Approximate Search...")
    hnsw_config = HNSWConfig(M=16, ef_construction=200, ef_search=64)
    vdb_hnsw = GrainVDB(dim=dim, mode=SearchMode.HNSW, hnsw_config=hnsw_config)
    vdb_hnsw.add_vectors(vectors)

    t0 = time.perf_counter()
    vdb_hnsw.build_index()
    hnsw_build_time = time.perf_counter() - t0
    print(f"  ✓ Built HNSW graph in {hnsw_build_time:.2f} s")

    stats = vdb_hnsw.hnsw_stats
    if stats:
        print(f"  ✓ Nodes: {stats.num_nodes:,} | Edges: {stats.num_edges:,} | Max Level: {stats.max_level}")

    hnsw_latencies = []
    recalls = []
    for i, q in enumerate(queries):
        res = vdb_hnsw.search(q, k=k)
        hnsw_latencies.append(res.latency_ms)
        gt = ground_truth_indices[i]
        matched = len(set(res.indices).intersection(gt))
        recalls.append(matched / float(k))

    hnsw_p50 = np.percentile(hnsw_latencies, 50)
    hnsw_p95 = np.percentile(hnsw_latencies, 95)
    mean_recall = np.mean(recalls)
    print(f"  ✓ Single Query Latency: p50: {hnsw_p50:.3f} ms | p95: {hnsw_p95:.3f} ms")
    print(f"  ✓ Recall@{k}: {mean_recall*100:.2f}% (vs exact scan)")

    # 4. Persistence & Page-Aligned Zero-Copy mmap
    print("\n[4/4] Benchmarking 4KB Page-Aligned Zero-Copy mmap...")
    tmp_path = "/tmp/grainvdb_benchmark.gvdb"
    t0 = time.perf_counter()
    vdb_cpu.save(tmp_path)
    save_time = (time.perf_counter() - t0) * 1000
    file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    print(f"  ✓ Saved {file_size_mb:.2f} MB page-aligned index in {save_time:.1f} ms")

    vdb_mmap = GrainVDB(dim=dim, mode=SearchMode.EXACT)
    t0 = time.perf_counter()
    vdb_mmap.mmap(tmp_path)
    mmap_time = (time.perf_counter() - t0) * 1000
    print(f"  ✓ Zero-copy mmap opened in {mmap_time:.2f} ms")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print()
    print("=" * 68)
    print("  Benchmark Summary & Architectural Comparison")
    print("=" * 68)
    print(f"  • CPU Accelerate Single-Query Latency (p50): {cpu_p50:.3f} ms ({cpu_p50*1000:.1f} µs) [100% recall]")
    print(f"  • Metal GPU Single-Query Latency (p50):      {gpu_p50:.3f} ms [100% recall]")
    print(f"  • Metal GPU Peak Batch Throughput:          {peak_qps:,.0f} queries/sec")
    print(f"  • HNSW Approximate Search Latency (p50):    {hnsw_p50:.3f} ms ({mean_recall*100:.1f}% recall)")
    print(f"  • Page-Aligned Zero-Copy mmap Load Time:    {mmap_time:.2f} ms")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GrainVDB Benchmark Runner")
    parser.add_argument("--n-vectors", type=int, default=20000, help="Number of vectors (default: 20000)")
    parser.add_argument("--dim", type=int, default=128, help="Vector dimension (default: 128)")
    parser.add_argument("--n-queries", type=int, default=50, help="Number of test queries (default: 50)")
    parser.add_argument("--k", type=int, default=10, help="Top-K results (default: 10)")
    args = parser.parse_args()

    run_benchmark(
        n_vectors=args.n_vectors,
        dim=args.dim,
        n_queries=args.n_queries,
        k=args.k,
    )
