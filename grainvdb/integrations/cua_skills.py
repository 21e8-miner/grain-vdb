"""
cua_skills.py — Chapter 5: Skills with Hands (Playbook Liveness Probes & Proactive Workflow Self-Repair).

Core Philosophy:
- Chapter 4 gave the agent "hands" to probe individual episodic memories.
- Chapter 5 elevates actuated curiosity to composite workflows: **Playbooks**.
- A Playbook represents a multi-step routine (e.g., "Export Weekly Accounting Report").
- Instead of discovering that a workflow is broken during a live scheduled execution,
  the mind runs **Playbook Liveness Probes** against step preconditions during idle cycles.
- If Step 1 or the environmental precondition fails an active probe, the playbook is
  automatically flagged as `DRIFTED` and triggers an autonomous self-healing plan.
"""

from __future__ import annotations

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .cua_hands import Level, Outcome, ProbingMind, ProbeExecutor, classify, ladder_level
from .cua import CuaGrainMemory

logger = logging.getLogger("cua.skills")


class PlaybookStatus(str, Enum):
    UNPROBED = "unprobed"
    HEALTHY = "healthy"
    DRIFTED = "drifted"
    NEEDS_REPAIR = "needs_repair"


@dataclass
class SkillStep:
    """Represents a single deterministic action step within a composite playbook."""
    step_id: int
    action_text: str
    target_app: Optional[str] = None
    precondition_text: Optional[str] = None
    precondition_embedding: Optional[List[float]] = None
    level_cap: Level = Level.INERT
    last_outcome: Optional[Outcome] = None


@dataclass
class SkillPlaybook:
    """A mined or user-authored multi-step automation workflow."""
    name: str
    description: str
    steps: List[SkillStep] = field(default_factory=list)
    status: PlaybookStatus = PlaybookStatus.UNPROBED
    trust_score: float = 1.0
    last_verified_ts: Optional[float] = None
    failure_step_index: Optional[int] = None
    repair_prompt: Optional[str] = None

    def add_step(
        self,
        action_text: str,
        target_app: Optional[str] = None,
        precondition_text: Optional[str] = None,
        precondition_embedding: Optional[List[float]] = None,
    ) -> SkillStep:
        verdict = classify(action_text)
        step = SkillStep(
            step_id=len(self.steps) + 1,
            action_text=action_text,
            target_app=target_app,
            precondition_text=precondition_text,
            precondition_embedding=precondition_embedding,
            level_cap=verdict.level
        )
        self.steps.append(step)
        return step

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "trust_score": self.trust_score,
            "last_verified_ts": self.last_verified_ts,
            "failure_step_index": self.failure_step_index,
            "repair_prompt": self.repair_prompt,
            "steps": [
                {
                    "step_id": s.step_id,
                    "action_text": s.action_text,
                    "target_app": s.target_app,
                    "precondition_text": s.precondition_text,
                    "level_cap": s.level_cap.name,
                    "last_outcome": s.last_outcome.name if s.last_outcome else None
                }
                for s in self.steps
            ]
        }


class PlaybookProber:
    """
    Actuated Liveness Engine for multi-step skills.
    Runs non-invasive precondition and entry-point probes to certify workflow health.
    """
    def __init__(
        self,
        mind: ProbingMind,
        playbooks: Optional[List[SkillPlaybook]] = None,
        liveness_interval_s: float = 7200.0,  # Probes playbooks every 2 hours
    ):
        self.mind = mind
        self.playbooks: Dict[str, SkillPlaybook] = {p.name: p for p in (playbooks or [])}
        self.liveness_interval_s = liveness_interval_s

    def register_playbook(self, playbook: SkillPlaybook) -> None:
        self.playbooks[playbook.name] = playbook

    async def probe_playbook_liveness(self, playbook_name: str) -> Dict[str, Any]:
        """
        Executes a targeted liveness probe on the entry-point (Step 1) of a playbook.
        Validates:
        1. Precondition screen match (L0 LOOK)
        2. Initial UI element presence (L1 PRESENCE)
        """
        pb = self.playbooks.get(playbook_name)
        if not pb:
            return {"error": f"Playbook '{playbook_name}' not found"}

        if not pb.steps:
            return {"error": "Playbook contains no steps"}

        now = time.time()
        step1 = pb.steps[0]
        
        # 1. Precondition Screen Check (Level 0 LOOK)
        if step1.precondition_embedding is not None and self.mind.executor is not None:
            live_screen = await self.mind.executor.screen_embedding()
            if len(live_screen) == len(step1.precondition_embedding):
                v0 = np.asarray(live_screen, dtype=np.float32)
                v1 = np.asarray(step1.precondition_embedding, dtype=np.float32)
                sim = float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1) + 1e-7))
                if sim < self.mind.look_lo:
                    pb.status = PlaybookStatus.DRIFTED
                    pb.trust_score = max(0.1, pb.trust_score * 0.5)
                    pb.failure_step_index = 0
                    pb.repair_prompt = f"Playbook '{pb.name}' failed L0 precondition probe: Live screen does not resemble launch state (sim={sim:.3f})."
                    step1.last_outcome = Outcome.REFUTED
                    return {
                        "playbook": pb.name,
                        "status": pb.status.value,
                        "probed_step": 1,
                        "failure_level": "L0_PRECONDITION",
                        "similarity": round(sim, 3),
                        "repair_prompt": pb.repair_prompt
                    }

        # 2. Entry-Point Element Presence Check (Level 1 PRESENCE)
        if self.mind.executor is not None:
            loc = await self.mind.executor.locate(step1.action_text)
            if loc is None:
                pb.status = PlaybookStatus.DRIFTED
                pb.trust_score = max(0.1, pb.trust_score * 0.5)
                pb.failure_step_index = 1
                pb.repair_prompt = f"Playbook '{pb.name}' failed L1 presence probe: Target element for '{step1.action_text}' is missing from live UI."
                step1.last_outcome = Outcome.REFUTED
                return {
                    "playbook": pb.name,
                    "status": pb.status.value,
                    "probed_step": 1,
                    "failure_level": "L1_PRESENCE",
                    "repair_prompt": pb.repair_prompt
                }

        # Certified Healthy
        pb.status = PlaybookStatus.HEALTHY
        pb.trust_score = min(1.0, pb.trust_score + 0.1)
        pb.last_verified_ts = now
        pb.failure_step_index = None
        pb.repair_prompt = None
        step1.last_outcome = Outcome.CONFIRMED

        return {
            "playbook": pb.name,
            "status": pb.status.value,
            "probed_step": 1,
            "trust_score": round(pb.trust_score, 2),
            "message": "Playbook entry-point verified healthy and ready for execution."
        }

    async def audit_all_playbooks(self) -> Dict[str, Any]:
        """Probes all registered playbooks and returns fleet health status."""
        results = {}
        for name in list(self.playbooks.keys()):
            results[name] = await self.probe_playbook_liveness(name)
        return results
