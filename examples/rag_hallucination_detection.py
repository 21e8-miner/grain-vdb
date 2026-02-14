#!/usr/bin/env python3
"""
RAG Hallucination Detection Example
Demonstrates using topology audit to detect potential hallucinations.
"""

import numpy as np
from grainvdb import GrainVDB, SearchMode, Quantization


def simulate_document_embeddings(n_docs: int, dim: int, n_topics: int) -> tuple:
    """
    Simulate document embeddings with topic clusters.
    Returns embeddings and topic labels.
    """
    rng = np.random.default_rng(42)
    
    # Create topic centers
    topic_centers = rng.standard_normal((n_topics, dim), dtype=np.float32)
    topic_centers /= np.linalg.norm(topic_centers, axis=1, keepdims=True) + 1e-12
    
    # Assign documents to topics
    topic_labels = rng.integers(0, n_topics, size=n_docs)
    
    # Generate documents around topic centers
    noise_level = 0.2
    embeddings = topic_centers[topic_labels] + noise_level * rng.standard_normal(
        (n_docs, dim), dtype=np.float32
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    
    return embeddings, topic_labels, topic_centers


def safe_retrieve(vdb, query, k: int = 5, coherence_threshold: float = 0.6):
    """
    Safely retrieve documents with hallucination detection.
    """
    result = vdb.search(query, k=k)
    audit = vdb.audit(result)
    
    response = {
        "indices": result.indices,
        "scores": result.scores,
        "latency_ms": result.latency_ms,
        "connectivity": audit.connectivity,
        "coherence": audit.coherence,
        "is_coherent": audit.is_semantically_coherent(coherence_threshold),
    }
    
    if not response["is_coherent"]:
        response["warning"] = "Low semantic coherence detected"
        response["suggestion"] = "Results may contain hallucinations. Try reformulating the query."
    
    return response


def main():
    print("=" * 70)
    print("RAG Hallucination Detection Example")
    print("=" * 70)
    
    # Configuration
    DIM = 384  # Typical embedding dimension (e.g., all-MiniLM-L6-v2)
    N_DOCS = 50_000
    N_TOPICS = 50
    
    print(f"\nConfiguration:")
    print(f"  Dimension: {DIM}")
    print(f"  Documents: {N_DOCS:,}")
    print(f"  Topics: {N_TOPICS}")
    
    # Initialize GrainVDB
    print("\n[1] Initializing GrainVDB...")
    vdb = GrainVDB(
        dim=DIM,
        mode=SearchMode.EXACT,
        quant=Quantization.FP16,
    )
    
    # Generate document embeddings
    print(f"[2] Generating {N_DOCS:,} document embeddings...")
    docs, labels, centers = simulate_document_embeddings(N_DOCS, DIM, N_TOPICS)
    vdb.add_vectors(docs)
    print(f"    Stored: {vdb.vector_count:,} documents")
    
    # Test 1: Coherent query (matches a topic)
    print("\n" + "-" * 70)
    print("Test 1: Coherent Query (matches topic 0)")
    print("-" * 70)
    
    coherent_query = centers[0] + 0.05 * np.random.randn(DIM).astype(np.float32)
    coherent_query /= np.linalg.norm(coherent_query) + 1e-12
    
    result = safe_retrieve(vdb, coherent_query, k=10)
    
    print(f"  Query: 'Tell me about topic 0' (simulated)")
    print(f"  Connectivity: {result['connectivity']:.4f}")
    print(f"  Coherence: {result['coherence']:.4f}")
    print(f"  Is Coherent: {result['is_coherent']}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    if result['is_coherent']:
        print("  ✅ PASS: Results are semantically coherent")
    else:
        print("  ⚠️  WARNING: Low coherence detected")
    
    # Test 2: Incoherent query (random noise)
    print("\n" + "-" * 70)
    print("Test 2: Incoherent Query (random noise)")
    print("-" * 70)
    
    random_query = np.random.randn(DIM).astype(np.float32)
    random_query /= np.linalg.norm(random_query) + 1e-12
    
    result = safe_retrieve(vdb, random_query, k=10)
    
    print(f"  Query: 'xyz abc 123 nonsense' (simulated)")
    print(f"  Connectivity: {result['connectivity']:.4f}")
    print(f"  Coherence: {result['coherence']:.4f}")
    print(f"  Is Coherent: {result['is_coherent']}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    
    if not result['is_coherent']:
        print("  ✅ PASS: Incoherence correctly detected")
        print(f"  Warning: {result.get('warning', 'N/A')}")
        print(f"  Suggestion: {result.get('suggestion', 'N/A')}")
    else:
        print("  ⚠️  WARNING: Incoherence not detected")
    
    # Test 3: Mixed query (partially coherent)
    print("\n" + "-" * 70)
    print("Test 3: Mixed Query (50% topic 1, 50% topic 2)")
    print("-" * 70)
    
    mixed_query = 0.5 * centers[1] + 0.5 * centers[2]
    mixed_query += 0.1 * np.random.randn(DIM).astype(np.float32)
    mixed_query /= np.linalg.norm(mixed_query) + 1e-12
    
    result = safe_retrieve(vdb, mixed_query, k=10, coherence_threshold=0.5)
    
    print(f"  Query: 'Compare topic 1 and topic 2' (simulated)")
    print(f"  Connectivity: {result['connectivity']:.4f}")
    print(f"  Coherence: {result['coherence']:.4f}")
    print(f"  Is Coherent: {result['is_coherent']}")
    
    if result['is_coherent']:
        print("  ✅ Results show moderate coherence")
    else:
        print("  ⚠️  Low coherence - results span multiple topics")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: Topology Audit for RAG")
    print("=" * 70)
    print("""
The topology audit measures the semantic coherence of retrieved results:

  - High connectivity (>0.7): Results are semantically related
    → Lower risk of hallucinations
    
  - Low connectivity (<0.4): Results are semantically fractured  
    → Higher risk of hallucinations
    
  - Coherence score combines connectivity and entropy
    → Use as confidence metric for RAG responses

Integration with LLM:
  1. Retrieve documents with GrainVDB
  2. Run topology audit on results
  3. If coherence < threshold:
     - Add warning to LLM prompt
     - Request clarification from user
     - Fall back to broader search
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
