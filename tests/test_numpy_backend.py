"""
test_numpy_backend.py — Unit tests for GrainVDB Pure NumPy Reference Engine.
"""

import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb import GrainVDB, SearchMode, EngineType, DistanceMetric


class TestNumpyBackend(unittest.TestCase):

    def setUp(self):
        self.dim = 32
        self.num_vectors = 100
        np.random.seed(42)
        self.vectors = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        self.metadata = [{"doc_id": i, "category": "tech" if i % 2 == 0 else "finance"} for i in range(self.num_vectors)]

    def test_numpy_crud_and_search(self):
        vdb = GrainVDB(dim=self.dim, mode=SearchMode.EXACT, engine=EngineType.NUMPY)
        self.assertEqual(vdb.engine, EngineType.NUMPY)

        # 1. Add vectors
        vdb.add_vectors(self.vectors, metadata=self.metadata)
        self.assertEqual(vdb.vector_count, self.num_vectors)

        # 2. Get vector
        vec_0 = vdb.get_vector(0)
        norm_v0 = self.vectors[0] / np.linalg.norm(self.vectors[0])
        np.testing.assert_allclose(vec_0, norm_v0, atol=1e-5)

        # 3. Exact search
        query = self.vectors[10].copy()
        res = vdb.search(query, k=3)
        self.assertEqual(res.indices[0], 10)
        self.assertAlmostEqual(res.scores[0], 1.0, places=4)

        # 4. Filtered search
        res_filtered = vdb.search(query, k=5, filter=lambda id, meta: meta.get("category") == "finance")
        for idx in res_filtered.indices:
            meta = vdb.get_metadata(idx)
            self.assertEqual(meta["category"], "finance")

        # 5. Update vector
        new_vec = np.zeros(self.dim, dtype=np.float32)
        new_vec[0] = 1.0
        vdb.update_vector(0, new_vec, metadata={"doc_id": 0, "category": "updated"})
        self.assertEqual(vdb.get_metadata(0)["category"], "updated")

        # 6. Remove vector
        vdb.remove_vectors([0])
        self.assertEqual(vdb.vector_count, self.num_vectors - 1)
        with self.assertRaises(KeyError):
            vdb.get_vector(0)

        # 7. Batch search
        batch_queries = self.vectors[15:20].copy()
        batch_res = vdb.search_batch(batch_queries, k=1)
        self.assertEqual(len(batch_res), 5)
        for i, r in enumerate(batch_res):
            self.assertEqual(r.indices[0], 15 + i)


if __name__ == "__main__":
    unittest.main()
