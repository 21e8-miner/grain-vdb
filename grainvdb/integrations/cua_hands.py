"""
cua_hands.py — Actuated Curiosity & Safe Active Probing for Computer Use Agents (CUAs).

Core Philosophy:
- Passive agent memory only reads audit trails (passive science).
- Actuated curiosity runs controlled micro-experiments to verify if recorded beliefs
  match live environmental reality before a human relies on them.

The IRB (Institutional Review Board in Code):
- R1: Allowlist Only — Unrecognized text defaults to Level 0 (LOOK: mirror probe, zero interaction).
- R2: The Ladder — Level N requires fresh evidence at Level N-1 within a time window.
- R3: Dynamic Drift Budget — Volatile environments loosen probe budget; calm ones throttle spend.
- R4: Quiescence — No probe executes if human activity is detected.
- R5: Replay, Never Invent — Plans compile strictly from recorded episode history.
- R6: Epistemic Quarantine — Probes update probe metrics/trust, never raw audit ground truth.
- Tri-State Safety Kernel: CONFIRMED / REFUTED / INCONCLUSIVE (all infra errors collapse to INCONCLUSIVE).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import os
import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union
import numpy as np

from .cua import CuaGrainMemory
from ..engine import GrainVDB, SearchMode, EngineType

logger = logging.getLogger("cua.hands")


# ============================================================================
# 1. THE LADDER & CLASSIFIER (R1 & R2)
# ============================================================================

class Level(IntEnum):
    LOOK = 0      # Zero interaction: screen embedding mirror comparison against GrainVDB
    PRESENCE = 1  # Locate element coordinates on screen (hover/highlight; no activation)
    INERT = 2     # Activate then immediately dismiss (open menu -> Escape; self-undoing)
    REPLAY = 3    # Full replay of idempotent allowlisted action with compensation


@dataclass(frozen=True)
class ProbeVerdict:
    level: Level
    reason: str


# Curated Deployment Allowlist (Order matters: most specific first)
_ALLOWLIST: Tuple[Tuple[re.Pattern[str], Level], ...] = (
    (re.compile(r"^click (menu|tab|dropdown|toggle) \S"), Level.INERT),
    (re.compile(r"^click (save|refresh|sort|collapse|expand)\b"), Level.REPLAY),
    (re.compile(r"^(hover|focus|select|highlight)\b"), Level.PRESENCE),
    (re.compile(r"^(open|view|read|show|display)\b"), Level.PRESENCE),
)

# Seatbelt Deny Gate (Fires before any allowlist match)
_FORBIDDEN = re.compile(
    r"\b(delete|remove|submit|send|pay|purchase|install|sudo|format|overwrite"
    r"|move|rename|share|logout|shutdown|mount|eject)\b"
)


def classify(text: str) -> ProbeVerdict:
    """R1: Enforces allowlist parsing with hard deny-gate overrides."""
    t = (text or "").strip().lower()
    if _FORBIDDEN.search(t):
        return ProbeVerdict(Level.LOOK, "forbidden verb detected")
    for pat, lv in _ALLOWLIST:
        if pat.match(t):
            return ProbeVerdict(lv, pat.pattern)
    return ProbeVerdict(Level.LOOK, "unrecognized action — LOOK only")


def ladder_level(cap: Level, achieved: int, fresh: bool) -> Level:
    """
    R2 Pure Function:
    - At most one rung above fresh evidence at Level N-1.
    - Never above the classifier's cap.
    - If evidence is expired or missing, drops down to LOOK.
    """
    if not fresh:
        achieved = -1
    return Level(min(int(cap), achieved + 1))


# ============================================================================
# 2. THREE-WAY OUTCOMES & DRIFT BUDGET (R3)
# ============================================================================

class Outcome(IntEnum):
    CONFIRMED = 0
    REFUTED = 1
    INCONCLUSIVE = 2


COSTS: Dict[Level, float] = {
    Level.LOOK: 0.5,
    Level.PRESENCE: 1.0,
    Level.INERT: 3.0,
    Level.REPLAY: 8.0
}


@dataclass
class ProbeBudget:
    """R3: Drift-sensitive budget economy."""
    base_per_cycle: float = 6.0
    drift_multiplier: float = 4.0
    cap: float = 0.0
    spent: float = 0.0

    def open_cycle(self, drift_ema: float) -> None:
        self.cap = self.base_per_cycle * (1.0 + self.drift_multiplier * drift_ema)
        self.spent = 0.0

    @property
    def remaining(self) -> float:
        return self.cap - self.spent

    def try_spend(self, cost: float) -> bool:
        if cost < 0 or (self.remaining + 1e-9) < cost:
            return False
        self.spent += cost
        return True


# ============================================================================
# 3. EXECUTOR & QUIESCENCE PROTOCOLS (R4)
# ============================================================================

class ProbeExecutor(Protocol):
    replay_supported: bool

    async def screen_embedding(self) -> List[float]: ...
    async def locate(self, semantic_text: str) -> Optional[Tuple[float, float]]: ...
    async def inert(self, semantic_text: str) -> bool: ...
    async def replay(self, semantic_text: str) -> bool: ...


class QuiescenceGuard(Protocol):
    async def is_quiescent(self) -> bool: ...


class AlwaysQuiescent:
    async def is_quiescent(self) -> bool:
        return True


class MockUIExecutor:
    """Mock execution harness for local unit testing and simulated UI verification."""
    def __init__(self, dim: int = 128, vanished_elements: Optional[set[str]] = None, replay_supported: bool = True):
        self.dim = dim
        self.vanished = vanished_elements or set()
        self.replay_supported = replay_supported

    async def screen_embedding(self) -> List[float]:
        return [0.1] * self.dim

    async def locate(self, semantic_text: str) -> Optional[Tuple[float, float]]:
        if semantic_text.lower() in self.vanished:
            return None
        return (100.0, 200.0)

    async def inert(self, semantic_text: str) -> bool:
        return semantic_text.lower() not in self.vanished

    async def replay(self, semantic_text: str) -> bool:
        return semantic_text.lower() not in self.vanished


# ============================================================================
# 4. EPISODIC LEDGER & TRUST STATE (R6 Epistemic Quarantine)
# ============================================================================

@dataclass
class EpisodeState:
    cua_seq: int
    text: str
    trust: float = 0.8
    outcome: Optional[bool] = None     # Ground truth audit outcome (Never modified by probes)
    probes: int = 0                    # Probes attempted on this episode
    probe_misses: int = 0              # Consecutive INCONCLUSIVE probe runs
    probe_level: int = 0               # Highest ladder rung with fresh evidence
    last_probed_ts: Optional[float] = None
    last_verified_ts: Optional[float] = None
    evicted: bool = False


class TrustLedger:
    """Thread-safe append-only episodic state ledger with epistemic quarantine."""
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self._states: Dict[int, EpisodeState] = {}
        self._lock = asyncio.Lock()

    def get(self, seq: int) -> Optional[EpisodeState]:
        return self._states.get(seq)

    def all_states(self) -> List[EpisodeState]:
        return list(self._states.values())

    def apply(self, ev: Dict[str, Any]) -> None:
        kind = ev.get("e")
        seq = int(ev.get("seq", 0))
        st = self._states.setdefault(seq, EpisodeState(cua_seq=seq, text=ev.get("text", "")))
        
        if kind == "w": # Write action
            st.text = ev.get("text", st.text)
        elif kind == "v": # Ground truth audit verification
            st.outcome = bool(ev.get("o"))
            st.trust = float(ev.get("t", st.trust))
            st.last_verified_ts = float(ev.get("ts", time.time()))
        elif kind == "p": # Probe measurement (EPISTEMIC QUARANTINE: Never touches st.outcome!)
            st.probes += 1
            st.last_probed_ts = float(ev["ts"])
            st.probe_level = max(st.probe_level, int(ev.get("lv", 0)))
            if ev.get("o") == "inconclusive":
                st.probe_misses += 1
            elif ev.get("o") in ("confirmed", "refuted") and "t" in ev:
                st.trust = float(ev["t"])
                st.probe_misses = 0

    def observe_probe(self, seq: int, level: int, outcome: Optional[str], trust: float, now: float) -> None:
        ev = {"e": "p", "seq": seq, "lv": int(level), "o": outcome, "t": trust, "ts": now}
        self.apply(ev)
        if self.path:
            try:
                with open(self.path, "a") as f:
                    f.write(json.dumps(ev) + "\n")
            except Exception as e:
                logger.warning("Failed to append probe event to ledger: %e", e)


# ============================================================================
# 5. PROBING MIND: ACTUATED CURIOSITY KERNEL
# ============================================================================

class ProbingMind:
    """
    Actuated curiosity engine bridging GrainVDB semantic visual search with
    the safe experimental probing ladder.
    """
    def __init__(
        self,
        dim: int = 128,
        memory: Optional[CuaGrainMemory] = None,
        executor: Optional[ProbeExecutor] = None,
        guard: Optional[QuiescenceGuard] = None,
        budget: Optional[ProbeBudget] = None,
        ledger_path: Optional[Union[str, Path]] = None,
        ladder_window_s: float = 3600.0,
        probe_timeout_s: float = 10.0,
        probe_confirm_rate: float = 0.25,
        probe_refute_rate: float = 0.50,
        look_confirm_threshold: float = 0.85,
        look_refute_threshold: float = 0.45,
        max_probe_misses: int = 3,
    ):
        self.dim = dim
        self.memory = memory or CuaGrainMemory(dim=dim, engine=EngineType.METAL)
        self.executor = executor
        self.guard = guard or AlwaysQuiescent()
        self.budget = budget or ProbeBudget()
        self.ledger = TrustLedger(Path(ledger_path) if ledger_path else None)
        
        self.ladder_window_s = ladder_window_s
        self.probe_timeout_s = probe_timeout_s
        self.probe_confirm = probe_confirm_rate
        self.probe_refute = probe_refute_rate
        self.look_hi = look_confirm_threshold
        self.look_lo = look_refute_threshold
        self.max_probe_misses = max_probe_misses
        
        self.drift_ema: float = 0.0
        self._probe_lock = asyncio.Lock()

    async def record_episode(self, seq: int, text: str, embedding: List[float], app_name: Optional[str] = None):
        """Records an episode in both GrainVDB and the trust ledger."""
        self.memory.record_action(
            cua_sequence_id=seq,
            semantic_text=text,
            screenshot_embedding=embedding,
            app_name=app_name
        )
        self.ledger.apply({"e": "w", "seq": seq, "text": text, "ts": time.time()})

    async def probe_cycle(self) -> Dict[str, Any]:
        """Runs a single explicit, opt-in active probing cycle across candidate memories."""
        if self.executor is None:
            return {"skipped": "no executor attached (LOOK-only mind)"}
        if not await self.guard.is_quiescent():
            return {"skipped": "machine busy (human active — R4)"}

        now = time.time()
        self.budget.open_cycle(self.drift_ema)
        
        candidates = [
            s for s in self.ledger.all_states()
            if not s.evicted and s.probe_misses < self.max_probe_misses
        ]

        report = {
            "planned": 0, "confirmed": 0, "refuted": 0,
            "inconclusive": 0, "blocked": 0, "spent": 0.0, "max_level": -1
        }

        async with self._probe_lock:
            for st in candidates:
                if self.budget.remaining <= 0:
                    break
                if not st.text:
                    continue

                verdict = classify(st.text)
                fresh = (st.last_probed_ts is not None and (now - st.last_probed_ts) <= self.ladder_window_s)
                level = ladder_level(verdict.level, st.probe_level, fresh)
                
                if level == Level.REPLAY and not getattr(self.executor, "replay_supported", False):
                    level = Level.INERT

                cost = COSTS[level]
                if not self.budget.try_spend(cost):
                    break

                report["planned"] += 1
                report["max_level"] = max(report["max_level"], int(level))

                try:
                    outcome = await asyncio.wait_for(
                        self._run_probe(st, level), timeout=self.probe_timeout_s
                    )
                except asyncio.TimeoutError:
                    outcome = Outcome.INCONCLUSIVE # Collapse infra failure safely
                except Exception as exc:
                    logger.warning("Probe execution failure at seq=%d: %r", st.cua_seq, exc)
                    outcome = Outcome.INCONCLUSIVE

                self._apply_probe(st.cua_seq, st, level, outcome)
                outcome_key = {Outcome.CONFIRMED: "confirmed", Outcome.REFUTED: "refuted", Outcome.INCONCLUSIVE: "inconclusive"}[outcome]
                report[outcome_key] += 1

        report["spent"] = round(self.budget.spent, 2)
        return report

    async def _run_probe(self, st: EpisodeState, level: Level) -> Outcome:
        if level == Level.LOOK:
            return await self._probe_look(st)
        assert self.executor is not None
        if level == Level.PRESENCE:
            loc = await self.executor.locate(st.text)
            return Outcome.CONFIRMED if loc is not None else Outcome.REFUTED
        if level == Level.INERT:
            ok = await self.executor.inert(st.text)
            return Outcome.CONFIRMED if ok else Outcome.REFUTED
        ok = await self.executor.replay(st.text)
        return Outcome.CONFIRMED if ok else Outcome.REFUTED

    async def _probe_look(self, st: EpisodeState) -> Outcome:
        """
        L0 Mirror Probe: Compares live screen vector against remembered GrainVDB neighborhood.
        Zero interaction, always eligible.
        """
        if self.executor is None:
            return Outcome.INCONCLUSIVE
            
        emb = await self.executor.screen_embedding()
        if len(emb) != self.dim:
            return Outcome.INCONCLUSIVE

        results = self.memory.semantic_recall(emb, k=5)
        if not results:
            return Outcome.REFUTED

        target_match = next((r for r in results if r["cua_sequence"] == st.cua_seq), None)
        best_score = max((r["similarity_score"] for r in results), default=0.0)

        if target_match and target_match["similarity_score"] >= self.look_hi:
            return Outcome.CONFIRMED
        if best_score < self.look_lo:
            return Outcome.REFUTED # Current screen looks completely foreign
        return Outcome.INCONCLUSIVE

    def _apply_probe(self, seq: int, st: EpisodeState, level: Level, outcome: Outcome) -> None:
        now = time.time()
        if outcome is Outcome.INCONCLUSIVE:
            # Infra timeout/error NEVER changes belief
            self.ledger.observe_probe(seq, int(level), "inconclusive", st.trust, now)
            return

        base_trust = st.trust
        if outcome is Outcome.CONFIRMED:
            new_trust = base_trust + self.probe_confirm * (1.0 - base_trust)
        else: # REFUTED
            new_trust = max(0.05, base_trust - self.probe_refute * base_trust)
            self.drift_ema = min(1.0, self.drift_ema + 0.25)

        self.ledger.observe_probe(seq, int(level), outcome.name.lower(), new_trust, now)
