#!/usr/bin/env python3
"""
GrainVDB Local-First RAG Demo
Demonstrates document indexing, metadata filtering, and semantic coherence audit.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb import GrainVDB, SearchMode, DistanceMetric


def main():
    print("=" * 60)
    print("  GrainVDB: Local-First RAG & Knowledge Store Demo")
    print("=" * 60)
    print()

    # Knowledge corpus
    documents = [
        {"title": "M2 Ultra Architecture", "topic": "hardware", "text": "Apple M2 Ultra features 24 CPU cores and up to 76 GPU cores with 800 GB/s unified memory bandwidth."},
        {"title": "Metal Shading Language", "topic": "graphics", "text": "Metal Shading Language provides low-overhead programmable GPU pipeline access on Apple Silicon."},
        {"title": "Unified Memory in RAG", "topic": "hardware", "text": "Unified memory enables zero-copy tensor sharing between CPU and Metal GPU without PCIe bus overhead."},
        {"title": "Commodity Crack Spreads", "topic": "finance", "text": "The 3:2:1 crack spread represents the theoretical margin of refining 3 barrels of crude oil into 2 barrels of gasoline and 1 barrel of distillate."},
        {"title": "VLCC Freight Tanker Rates", "topic": "finance", "text": "Baltic TD3C tanker route rates measure dirty tanker freight between the Middle East Gulf and China."},
        {"title": "Channel State Information", "topic": "rf_sensing", "text": "WiFi CSI captures subcarrier amplitude and phase perturbations caused by human motion and vital signs."},
    ]

    dim = 128
    np.random.seed(1337)

    # Simulate realistic semantic topic embeddings
    topic_vectors = {
        "hardware": np.random.randn(dim).astype(np.float32),
        "graphics": np.random.randn(dim).astype(np.float32),
        "finance": np.random.randn(dim).astype(np.float32),
        "rf_sensing": np.random.randn(dim).astype(np.float32),
    }

    doc_vectors = []
    for doc in documents:
        # Vector = topic vector + slight document variance
        base = topic_vectors[doc["topic"]]
        noise = np.random.randn(dim).astype(np.float32) * 0.2
        v = base + noise
        v = v / np.linalg.norm(v)
        doc_vectors.append(v)

    doc_vectors = np.array(doc_vectors, dtype=np.float32)

    # Initialize GrainVDB
    print("[1] Initializing in-process GrainVDB (Exact Metal Scan)...")
    vdb = GrainVDB(dim=dim, mode=SearchMode.EXACT, distance=DistanceMetric.COSINE)
    vdb.add_vectors(doc_vectors, metadata=documents)
    print(f"  ✓ Ingested {vdb.vector_count} knowledge chunks into Metal shared memory\n")

    # 1. Semantic Search
    print("[2] Running Semantic Query: 'Apple Silicon GPU and Memory Bandwidth'...")
    query_hardware = topic_vectors["hardware"] + np.random.randn(dim).astype(np.float32) * 0.1
    query_hardware /= np.linalg.norm(query_hardware)

    results = vdb.search(query_hardware, k=3)
    for rank, (idx, score, meta) in enumerate(zip(results.indices, results.scores, results.metadata), start=1):
        print(f"  [{rank}] Score: {score:.4f} | [{meta['topic'].upper()}] {meta['title']}")
        print(f"      Excerpt: \"{meta['text']}\"")

    # 2. Filtered Search
    print("\n[3] Running Metadata Filtered Query (Topic = 'finance')...")
    results_filtered = vdb.search(
        query_hardware,  # query is hardware, but filter requests finance
        k=2,
        filter=lambda vid, meta: meta.get("topic") == "finance",
    )
    for rank, (idx, score, meta) in enumerate(zip(results_filtered.indices, results_filtered.scores, results_filtered.metadata), start=1):
        print(f"  [{rank}] Score: {score:.4f} | [{meta['topic'].upper()}] {meta['title']}")
        print(f"      Excerpt: \"{meta['text']}\"")

    # 3. Topology Audit for Semantic Coherence
    print("\n[4] Running Topology Audit (Hallucination / Fracture Detection)...")
    audit = vdb.audit(results)
    print(f"  ✓ Connectivity: {audit.connectivity:.2f} (Pairs > 0.85 similarity)")
    print(f"  ✓ Coherence:    {audit.coherence:.2f}")
    print(f"  ✓ Semantic Coherence Status: {'✅ Coherent (Low Hallucination Risk)' if audit.is_semantically_coherent(0.5) else '⚠️ Semantic Fracture Detected'}")

    print("\n============================================================")
    print("  Demo Complete!")
    print("============================================================")


if __name__ == "__main__":
    main()
