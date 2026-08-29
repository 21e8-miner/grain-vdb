"""
test_merkle.py — Unit tests for Merkle-DAG Trajectory Chaining and Temporal Slicing.
"""

import os
import sys
import unittest
import numpy as np
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb.integrations.cua_merkle import MerkleTrajectoryChain, compute_vector_digest
from grainvdb.integrations.cua import CuaGrainMemory
from grainvdb.engine import EngineType


class TestMerkleTrajectory(unittest.TestCase):

    def test_vector_digest_deterministic(self):
        """Test vector hashing produces identical fingerprints across calls."""
        v1 = [0.1234567, 0.9876543, -0.5555555]
        v2 = [0.1234567, 0.9876543, -0.5555555]
        d1 = compute_vector_digest(v1)
        d2 = compute_vector_digest(v2)
        self.assertEqual(d1, d2)

    def test_merkle_chain_creation_and_integrity(self):
        """Test appending steps creates continuous, mathematically verifiable chain."""
        chain = MerkleTrajectoryChain(session_id="test-session-001")
        self.assertEqual(chain.root_hash, MerkleTrajectoryChain.GENESIS_HASH)

        dim = 8
        n1 = chain.append_step(1, "open Finder", [0.1] * dim)
        self.assertEqual(n1.parent_hash, MerkleTrajectoryChain.GENESIS_HASH)
        self.assertEqual(chain.root_hash, n1.node_hash)

        n2 = chain.append_step(2, "click Downloads", [0.2] * dim)
        self.assertEqual(n2.parent_hash, n1.node_hash)
        self.assertEqual(chain.root_hash, n2.node_hash)

        n3 = chain.append_step(3, "select file.pdf", [0.3] * dim)
        self.assertEqual(n3.parent_hash, n2.node_hash)
        self.assertEqual(chain.length, 3)

        # Verify integrity
        valid, err = chain.verify_integrity()
        self.assertTrue(valid)
        self.assertIsNone(err)

    def test_merkle_tamper_detection(self):
        """Test that altering any historical action or vector immediately trips verification."""
        chain = MerkleTrajectoryChain()
        dim = 8
        chain.append_step(1, "open Safari", [0.1] * dim)
        chain.append_step(2, "navigate to bank.com", [0.2] * dim)
        chain.append_step(3, "submit transfer $1000", [0.3] * dim)

        valid, _ = chain.verify_integrity()
        self.assertTrue(valid)

        # Tamper with step 2 action payload in memory
        chain.chain[1].action_payload = "navigate to phishing.com"
        valid, err = chain.verify_integrity()
        self.assertFalse(valid)
        self.assertIn("Tampered content at step 2", err)

    def test_merkle_inclusion_proof(self):
        """Test generating audit-grade cryptographic proof for a specific step."""
        chain = MerkleTrajectoryChain()
        dim = 8
        for i in range(1, 6):
            chain.append_step(i, f"Action #{i}", [float(i)*0.1] * dim)

        proof = chain.get_merkle_proof(3)
        self.assertEqual(proof["sequence_id"], 3)
        self.assertEqual(proof["ancestor_count"], 2)
        self.assertEqual(proof["total_chain_length"], 5)
        self.assertEqual(proof["tamper_evident_status"], "VERIFIED_VALID")

    def test_cua_grain_memory_temporal_trajectory_window(self):
        """Test recording multi-step sub-trajectories with exponential decay."""
        dim = 16
        mem = CuaGrainMemory(dim=dim, engine=EngineType.ACCELERATE)

        # Simulate 3-step sub-trajectory
        steps = [
            {"embedding": [0.1] * dim, "action": "Click Dropdown"},
            {"embedding": [0.2] * dim, "action": "Select Option A"},
            {"embedding": [0.3] * dim, "action": "Confirm Selection"},
        ]

        success = mem.record_trajectory_window(lead_sequence_id=42, steps=steps, decay=0.75)
        self.assertTrue(success)
        self.assertEqual(mem.total_records, 1)

        # Search for trajectory
        recalled = mem.semantic_recall(query_embedding=[0.25] * dim, k=1)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["cua_sequence"], 42)
        self.assertIn("Trajectory Window", recalled[0]["semantic_context"])

        # Check Merkle Proof on memory instance
        proof = mem.get_merkle_proof(42)
        self.assertEqual(proof["sequence_id"], 42)
        self.assertEqual(proof["tamper_evident_status"], "VERIFIED_VALID")

        mem.close()


if __name__ == "__main__":
    unittest.main()
