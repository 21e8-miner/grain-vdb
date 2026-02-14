#!/usr/bin/env python3
"""
GrainVDB Basic Search Example
Demonstrates fundamental vector search operations.
"""

import numpy as np
from grainvdb import GrainVDB, SearchMode, Quantization


def main():
    print("=" * 60)
    print("GrainVDB Basic Search Example")
    print("=" * 60)
    
    # Configuration
    DIM = 128
    N_VECTORS = 100_000
    N_QUERIES = 10
    K = 5
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {DIM}")
    print(f"  Vectors: {N_VECTORS:,}")
    print(f"  Queries: {N_QUERIES}")
    print(f"  K: {K}")
    
    # Initialize GrainVDB
    print("\n[1] Initializing GrainVDB...")
    vdb = GrainVDB(
        dim=DIM,
        mode=SearchMode.EXACT,
        quant=Quantization.FP16,
    )
    
    # Generate random vectors
    print(f"[2] Generating {N_VECTORS:,} random vectors...")
    vectors = np.random.randn(N_VECTORS, DIM).astype(np.float32)
    
    # Add vectors to database
    print("[3] Adding vectors to database...")
    vdb.add_vectors(vectors)
    print(f"    Stored: {vdb.vector_count:,} vectors")
    
    # Warmup GPU
    print("[4] Warming up GPU...")
    vdb.warmup()
    
    # Generate queries
    print(f"[5] Generating {N_QUERIES} query vectors...")
    queries = np.random.randn(N_QUERIES, DIM).astype(np.float32)
    
    # Single query search
    print("\n[6] Single Query Search:")
    for i, query in enumerate(queries[:3]):
        result = vdb.search(query, k=K)
        print(f"    Query {i+1}: {result.latency_ms:.2f}ms")
        print(f"      Top match score: {result.scores[0]:.4f}")
    
    # Batch query search (much faster!)
    print("\n[7] Batch Query Search:")
    import time
    start = time.time()
    results = vdb.search_batch(queries, k=K)
    batch_time = (time.time() - start) * 1000
    print(f"    Total time: {batch_time:.2f}ms")
    print(f"    Per query: {batch_time/N_QUERIES:.2f}ms")
    print(f"    Throughput: {N_QUERIES*1000/batch_time:.0f} QPS")
    
    # Topology audit
    print("\n[8] Topology Audit (Semantic Coherence):")
    result = vdb.search(queries[0], k=10)
    audit = vdb.audit(result)
    print(f"    Connectivity: {audit.connectivity:.4f}")
    print(f"    Coherence: {audit.coherence:.4f}")
    print(f"    Is coherent: {audit.is_semantically_coherent()}")
    
    # Performance metrics
    print("\n[9] Performance Metrics:")
    metrics = vdb.get_metrics()
    print(f"    Total queries: {metrics.total_queries}")
    print(f"    Avg latency: {metrics.avg_latency_ms:.2f}ms")
    print(f"    P95 latency: {metrics.p95_latency_ms:.2f}ms")
    print(f"    Throughput: {metrics.throughput_qps:.0f} QPS")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
