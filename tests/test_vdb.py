"""
GrainVDB Test Suite
Tests for Apple Silicon Metal-accelerated vector store.
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb import (
    GrainVDB,
    SearchMode,
    Quantization,
    DistanceMetric,
    EngineType,
    HNSWConfig,
)


class TestGrainVDB(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.dim = 128
        self.num_vectors = 500
        # Generate normalized random vectors
        raw = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        self.vectors = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.metadata = [{"doc_id": i, "category": "tech" if i % 2 == 0 else "finance"} for i in range(self.num_vectors)]

    def test_exact_search_recall(self):
        """Test exact brute force search returns highest similarity match."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors, metadata=self.metadata)
        self.assertEqual(vdb.vector_count, self.num_vectors)

        # Query with the exact first vector
        query = self.vectors[0]
        result = vdb.search(query, k=5)

        self.assertEqual(len(result.indices), 5)
        # Top result must be index 0 with similarity ~1.0
        self.assertEqual(result.indices[0], 0)
        self.assertAlmostEqual(result.scores[0], 1.0, places=2)
        self.assertEqual(result.metadata[0]["doc_id"], 0)

    def test_batch_search(self):
        """Test batch query processing."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors, metadata=self.metadata)

        num_queries = 10
        queries = self.vectors[:num_queries]
        results = vdb.search_batch(queries, k=5)

        self.assertEqual(len(results), num_queries)
        for i in range(num_queries):
            self.assertEqual(results[i].indices[0], i)
            self.assertAlmostEqual(results[i].scores[0], 1.0, places=2)

    def test_hnsw_search(self):
        """Test HNSW approximate search build and query."""
        hnsw_cfg = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.HNSW, hnsw_config=hnsw_cfg)
        vdb.add_vectors(self.vectors, metadata=self.metadata)
        vdb.build_index()

        stats = vdb.hnsw_stats
        self.assertIsNotNone(stats)
        self.assertGreater(stats.num_nodes, 0)
        self.assertGreater(stats.num_edges, 0)

        # Query with first vector
        query = self.vectors[0]
        result = vdb.search(query, k=5)
        self.assertEqual(len(result.indices), 5)
        self.assertEqual(result.indices[0], 0)

    def test_metadata_filtering(self):
        """Test search with predicate metadata filtering."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors, metadata=self.metadata)

        # Filter only finance category (odd indices)
        query = self.vectors[0]  # doc 0 is tech
        result = vdb.search(query, k=5, filter=lambda vid, meta: meta.get("category") == "finance")

        self.assertGreater(len(result.indices), 0)
        for meta in result.metadata:
            self.assertEqual(meta["category"], "finance")

    def test_vector_crud(self):
        """Test vector get, update, and remove."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors, metadata=self.metadata)

        # 1. Get vector
        vec_0 = vdb.get_vector(0)
        np.testing.assert_allclose(vec_0, self.vectors[0], atol=1e-2)

        # 2. Update vector
        new_vec = np.zeros(self.dim, dtype=np.float32)
        new_vec[0] = 1.0
        vdb.update_vector(0, new_vec, metadata={"doc_id": 0, "category": "updated"})
        updated_vec = vdb.get_vector(0)
        np.testing.assert_allclose(updated_vec, new_vec, atol=1e-2)

        # 3. Remove vector
        initial_count = vdb.vector_count
        vdb.remove_vectors([0])
        self.assertEqual(vdb.vector_count, initial_count - 1)

        with self.assertRaises(KeyError):
            vdb.get_vector(0)

    def test_persistence_save_load(self):
        """Test index save and load from disk."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors, metadata=self.metadata)

        with tempfile.NamedTemporaryFile(suffix=".gvdb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.assertTrue(vdb.save(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)

            # Load into new database instance
            vdb_loaded = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
            self.assertTrue(vdb_loaded.load(tmp_path))
            self.assertEqual(vdb_loaded.vector_count, self.num_vectors)

            # Query loaded database
            result = vdb_loaded.search(self.vectors[10], k=3)
            self.assertEqual(result.indices[0], 10)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_mmap_persistence(self):
        """Test memory-mapped zero-copy index loading."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors)

        with tempfile.NamedTemporaryFile(suffix=".gvdb", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.assertTrue(vdb.save(tmp_path))

            vdb_mmap = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
            self.assertTrue(vdb_mmap.mmap(tmp_path))
            self.assertEqual(vdb_mmap.vector_count, self.num_vectors)

            result = vdb_mmap.search(self.vectors[5], k=3)
            self.assertEqual(result.indices[0], 5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_topology_audit(self):
        """Test semantic coherence audit."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        vdb.add_vectors(self.vectors)

        # Audit a tight cluster (query against itself)
        result = vdb.search(self.vectors[0], k=5)
        audit = vdb.audit(result)
        self.assertIsInstance(audit.connectivity, float)
        self.assertIsInstance(audit.coherence, float)

    def test_engine_switching(self):
        """Test dynamic switching between Accelerate and Metal engines."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT, engine=EngineType.ACCELERATE)
        vdb.add_vectors(self.vectors)
        self.assertEqual(vdb.engine, EngineType.ACCELERATE)

        res_cpu = vdb.search(self.vectors[0], k=5)
        self.assertEqual(res_cpu.indices[0], 0)

        vdb.engine = EngineType.METAL
        self.assertEqual(vdb.engine, EngineType.METAL)

        res_gpu = vdb.search(self.vectors[0], k=5)
        self.assertEqual(res_gpu.indices[0], 0)

    def test_langchain_vectorstore(self):
        """Test LangChain vector store adapter."""
        from grainvdb.integrations import GrainVDBVectorStore

        # Mock embedding model
        class MockEmbedding:
            def embed_documents(self, texts):
                np.random.seed(42)
                return [np.random.randn(32).astype(np.float32) for _ in texts]

            def embed_query(self, text):
                np.random.seed(42)
                return np.random.randn(32).astype(np.float32)

        vs = GrainVDBVectorStore(embedding=MockEmbedding(), dim=32)
        docs = ["Apple Silicon M2 Ultra", "Neural Engine CoreML", "Local RAG Vector Store"]
        metas = [{"topic": "hardware"}, {"topic": "ai"}, {"topic": "database"}]

        ids = vs.add_texts(docs, metadatas=metas)
        self.assertEqual(len(ids), 3)

        results = vs.similarity_search("M2 Ultra hardware", k=2)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("topic", results[0].metadata)


    def test_edge_cases_empty_and_single_vector(self):
        """Test search on empty database and single vector database."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        # Empty DB search should fail gracefully
        with self.assertRaises(RuntimeError):
            vdb.search(self.vectors[0], k=3)

        # Ingest single vector
        vdb.add_vectors(self.vectors[:1])
        self.assertEqual(vdb.vector_count, 1)
        res = vdb.search(self.vectors[0], k=1)
        self.assertEqual(len(res.indices), 1)
        self.assertEqual(res.indices[0], 0)

    def test_edge_cases_dimension_mismatch(self):
        """Test ValueError on dimension mismatch."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        wrong_dim_vec = np.random.randn(self.dim + 4).astype(np.float32)
        with self.assertRaises(ValueError):
            vdb.add_vectors(np.array([wrong_dim_vec]))
        with self.assertRaises(ValueError):
            vdb.search(wrong_dim_vec, k=1)

    def test_edge_cases_zero_vector(self):
        """Test zero-magnitude vector handling without NaN/Inf."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT)
        zero_vec = np.zeros(self.dim, dtype=np.float32)
        vdb.add_vectors(np.array([zero_vec]))
        res = vdb.search(zero_vec, k=1)
        self.assertEqual(len(res.indices), 1)
        self.assertFalse(np.isnan(res.scores[0]))

    def test_hnsw_parameter_tuning(self):
        """Test HNSW runtime ef_search adjustment."""
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.HNSW)
        vdb.add_vectors(self.vectors)
        vdb.build_index()
        self.assertTrue(vdb.set_ef_search(128))
        res = vdb.search(self.vectors[0], k=5)
        self.assertGreaterEqual(len(res.indices), 1)


if __name__ == "__main__":
    unittest.main()
