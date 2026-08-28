"""
test_hands.py — Comprehensive test suite for Chapter 4: Actuated Curiosity & IRB Enforcements.
"""

import os
import sys
import unittest
import asyncio
import time
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from grainvdb.integrations.cua_hands import (
    Level,
    Outcome,
    ProbeBudget,
    ProbeVerdict,
    classify,
    ladder_level,
    TrustLedger,
    EpisodeState,
    ProbingMind,
    MockUIExecutor,
    COSTS,
)


class TestHandsIRB(unittest.IsolatedAsyncioTestCase):
    
    # -------------------------------------------------------------------------
    # R1: The Seatbelt & Allowlist Classification
    # -------------------------------------------------------------------------
    def test_r1_forbidden_verbs_never_escalate(self):
        """R1: Any action containing a dangerous/irreversible verb is hard-clamped to LOOK."""
        forbidden_samples = [
            "delete file report.pdf",
            "sudo rm -rf /",
            "click remove user",
            "submit payment form",
            "send email to team",
            "pay invoice #123",
            "purchase item in cart",
            "install malicious-pkg",
            "format hard drive",
            "overwrite database.sql",
            "move sensitive_folder to trash",
            "rename admin_user",
            "share screen with external",
            "logout current session",
            "shutdown host machine",
        ]
        for text in forbidden_samples:
            verdict = classify(text)
            self.assertEqual(verdict.level, Level.LOOK, f"Forbidden text '{text}' escalated to {verdict.level.name}!")

    def test_r1_allowlist_classification_shapes(self):
        """R1: Allowlisted actions classify into their exact intended maximum rung."""
        test_cases = [
            ("click menu File", Level.INERT),
            ("click tab Settings", Level.INERT),
            ("click dropdown Options", Level.INERT),
            ("click toggle DarkMode", Level.INERT),
            ("click save", Level.REPLAY),
            ("click refresh", Level.REPLAY),
            ("click sort", Level.REPLAY),
            ("hover over toolbar icon", Level.PRESENCE),
            ("focus search input", Level.PRESENCE),
            ("open browser window", Level.PRESENCE),
            ("view document details", Level.PRESENCE),
            ("click savefile", Level.LOOK),  # \b word boundary guard test
            ("unrecognized gibberish action", Level.LOOK),
        ]
        for text, expected_level in test_cases:
            verdict = classify(text)
            self.assertEqual(verdict.level, expected_level, f"Text '{text}' classified as {verdict.level.name}, expected {expected_level.name}")

    # -------------------------------------------------------------------------
    # R2: The Ladder Progression & Evidence Expiry
    # -------------------------------------------------------------------------
    def test_r2_ladder_progression_pure_function(self):
        """R2: Ladder cannot jump rungs, requires fresh evidence, and drops to LOOK on stale clearance."""
        cap = Level.REPLAY

        # Cold / no evidence -> must start at LOOK (Level 0)
        self.assertEqual(ladder_level(cap=cap, achieved=-1, fresh=False), Level.LOOK)
        self.assertEqual(ladder_level(cap=cap, achieved=0, fresh=False), Level.LOOK)

        # Fresh evidence at Level 0 -> can step to Level 1
        self.assertEqual(ladder_level(cap=cap, achieved=0, fresh=True), Level.PRESENCE)

        # Fresh evidence at Level 1 -> can step to Level 2
        self.assertEqual(ladder_level(cap=cap, achieved=1, fresh=True), Level.INERT)

        # Fresh evidence at Level 2 -> can step to Level 3 (REPLAY)
        self.assertEqual(ladder_level(cap=cap, achieved=2, fresh=True), Level.REPLAY)

        # Cap clamps higher achieved levels
        self.assertEqual(ladder_level(cap=Level.INERT, achieved=2, fresh=True), Level.INERT)

    # -------------------------------------------------------------------------
    # R3: Drift-Sensitive Budget Economy
    # -------------------------------------------------------------------------
    def test_r3_probe_budget_never_overdraws(self):
        """R3: Budget never overdraws and expands with drift."""
        budget = ProbeBudget(base_per_cycle=6.0, drift_multiplier=4.0)

        # Zero drift cycle (cap = 6.0)
        budget.open_cycle(drift_ema=0.0)
        self.assertEqual(budget.cap, 6.0)
        self.assertTrue(budget.try_spend(COSTS[Level.INERT]))  # 3.0 spend -> 3.0 remain
        self.assertTrue(budget.try_spend(COSTS[Level.PRESENCE]))  # 1.0 spend -> 2.0 remain
        self.assertFalse(budget.try_spend(COSTS[Level.INERT]))  # 3.0 spend > 2.0 remaining (Rejected)
        self.assertEqual(budget.spent, 4.0)
        self.assertEqual(budget.remaining, 2.0)

        # High drift cycle (cap = 6.0 * (1 + 4.0 * 0.5) = 18.0)
        budget.open_cycle(drift_ema=0.5)
        self.assertEqual(budget.cap, 18.0)
        self.assertTrue(budget.try_spend(COSTS[Level.REPLAY]))  # 8.0 spend
        self.assertTrue(budget.try_spend(COSTS[Level.REPLAY]))  # 8.0 spend
        self.assertFalse(budget.try_spend(COSTS[Level.INERT]))  # 3.0 > 2.0 (Rejected)
        self.assertEqual(budget.spent, 16.0)

    # -------------------------------------------------------------------------
    # R4: Quiescence Protection
    # -------------------------------------------------------------------------
    async def test_r4_quiescence_aborts_active_probing(self):
        """R4: Probing immediately skips if machine is not quiescent (human active)."""
        class BusyGuard:
            async def is_quiescent(self) -> bool:
                return False

        mind = ProbingMind(dim=8, guard=BusyGuard(), executor=MockUIExecutor(8))
        await mind.record_episode(1, "click menu File", [0.1] * 8)
        report = await mind.probe_cycle()
        self.assertIn("skipped", report)
        self.assertIn("machine busy", report["skipped"])

    # -------------------------------------------------------------------------
    # R6: Epistemic Quarantine & Tri-State Safety Kernel
    # -------------------------------------------------------------------------
    def test_r6_epistemic_quarantine_inconclusive_touches_nothing(self):
        """R6: INCONCLUSIVE outcomes (infra errors) must NEVER shift belief or touch audit outcome."""
        ledger = TrustLedger()
        
        # Write step and record ground truth audit verification
        ledger.apply({"e": "w", "seq": 42, "text": "click save", "ts": 1.0})
        ledger.apply({"e": "v", "seq": 42, "o": True, "t": 0.85, "ts": 2.0})

        st = ledger.get(42)
        self.assertIsNotNone(st)
        self.assertTrue(st.outcome)
        self.assertEqual(st.trust, 0.85)

        # Apply a failed/inconclusive probe carrying a poisoned trust value
        ledger.observe_probe(seq=42, level=int(Level.INERT), outcome="inconclusive", trust=0.01, now=3.0)

        st = ledger.get(42)
        # 1. Trust must remain untouched at 0.85
        self.assertEqual(st.trust, 0.85)
        # 2. Audit ground truth verdict must remain untouched
        self.assertTrue(st.outcome)
        # 3. Probe misses must increment
        self.assertEqual(st.probe_misses, 1)
        self.assertEqual(st.probes, 1)

    # -------------------------------------------------------------------------
    # End-to-End Active Probing & L0 Mirror Probe Cycle
    # -------------------------------------------------------------------------
    async def test_end_to_end_active_probing_cycle(self):
        """Simulates full probing cycle with element vanishing and refutation."""
        executor = MockUIExecutor(dim=8, vanished_elements={"click menu export", "click save"})
        mind = ProbingMind(dim=8, executor=executor, ladder_window_s=10.0)

        # Record episodes
        await mind.record_episode(1, "hover over toolbar", [0.1] * 8)
        await mind.record_episode(2, "click menu export", [0.2] * 8)
        await mind.record_episode(3, "click save", [0.3] * 8)

        # Initial probe cycle (fresh evidence, progresses from LOOK to PRESENCE / INERT)
        report = await mind.probe_cycle()
        self.assertGreater(report["planned"], 0)
        self.assertGreater(report["spent"], 0.0)

        st_vanished = mind.ledger.get(2)
        self.assertIsNotNone(st_vanished)
        self.assertGreater(st_vanished.probes, 0)


if __name__ == "__main__":
    unittest.main()
