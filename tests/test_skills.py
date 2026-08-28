"""
test_skills.py — Unit tests for Chapter 5: Skills with Hands (Playbook Liveness Probes).
"""

import os
import sys
import unittest
import asyncio
import time

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb.integrations.cua_hands import (
    Level,
    Outcome,
    ProbingMind,
    MockUIExecutor,
)
from grainvdb.integrations.cua_skills import (
    SkillPlaybook,
    SkillStep,
    PlaybookProber,
    PlaybookStatus,
)


class TestSkillsPlaybookProber(unittest.IsolatedAsyncioTestCase):

    def test_playbook_creation_and_step_allowlist(self):
        """Test that playbook steps automatically inherit correct classifier level caps."""
        pb = SkillPlaybook(name="Weekly Report", description="Export quarterly financial statement")
        step1 = pb.add_step("open Finder", target_app="Finder")
        step2 = pb.add_step("click menu Export", target_app="Excel")
        step3 = pb.add_step("click save", target_app="Excel")
        step4 = pb.add_step("delete temporary draft", target_app="Finder")

        self.assertEqual(len(pb.steps), 4)
        self.assertEqual(step1.level_cap, Level.PRESENCE)
        self.assertEqual(step2.level_cap, Level.INERT)
        self.assertEqual(step3.level_cap, Level.REPLAY)
        self.assertEqual(step4.level_cap, Level.LOOK) # Deny-gate clamped to LOOK!

    async def test_playbook_liveness_healthy(self):
        """Test that healthy playbook passes liveness probes and receives certified status."""
        dim = 8
        executor = MockUIExecutor(dim=dim)
        mind = ProbingMind(dim=dim, executor=executor)
        prober = PlaybookProber(mind=mind)

        pb = SkillPlaybook(name="Healthy Export", description="Automated export routine")
        pb.add_step("open Finder", target_app="Finder", precondition_embedding=[0.1] * dim)
        pb.add_step("click save", target_app="Finder")
        prober.register_playbook(pb)

        res = await prober.probe_playbook_liveness("Healthy Export")
        self.assertEqual(res["status"], PlaybookStatus.HEALTHY.value)
        self.assertEqual(pb.status, PlaybookStatus.HEALTHY)
        self.assertIsNone(pb.repair_prompt)
        self.assertIsNotNone(pb.last_verified_ts)

    async def test_playbook_liveness_drift_detected(self):
        """Test that when entry-point element is missing, playbook is marked DRIFTED with self-healing prompt."""
        dim = 8
        # MockUIExecutor with vanished entry point button
        executor = MockUIExecutor(dim=dim, vanished_elements={"click menu reports"})
        mind = ProbingMind(dim=dim, executor=executor)
        prober = PlaybookProber(mind=mind)

        pb = SkillPlaybook(name="Drifted Reports", description="Accounting reports")
        pb.add_step("click menu reports", target_app="QuickBooks")
        prober.register_playbook(pb)

        res = await prober.probe_playbook_liveness("Drifted Reports")
        self.assertEqual(res["status"], PlaybookStatus.DRIFTED.value)
        self.assertEqual(pb.status, PlaybookStatus.DRIFTED)
        self.assertIn("failed L1 presence probe", res["repair_prompt"])
        self.assertIn("repair_prompt", res)

    async def test_playbook_liveness_l0_precondition_drift(self):
        """Test that mismatched precondition screen triggers L0 drift alert."""
        dim = 8
        # Executor screen differs from expected precondition embedding
        executor = MockUIExecutor(dim=dim)
        mind = ProbingMind(dim=dim, executor=executor)
        prober = PlaybookProber(mind=mind)

        pb = SkillPlaybook(name="Photoshop Batch", description="Process images")
        # Precondition vector pointing in completely opposite direction
        opposite_precondition = [-0.5 if j % 2 == 0 else 0.5 for j in range(dim)]
        pb.add_step("click menu Filter", precondition_embedding=opposite_precondition)
        prober.register_playbook(pb)

        res = await prober.probe_playbook_liveness("Photoshop Batch")
        self.assertEqual(res["status"], PlaybookStatus.DRIFTED.value)
        self.assertEqual(res["failure_level"], "L0_PRECONDITION")


if __name__ == "__main__":
    unittest.main()
