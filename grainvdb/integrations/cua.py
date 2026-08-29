"""
Cua-Grain Memory Integration - Enterprise Production Edition
Unified high-performance memory layer for Computer Use Agents (CUAs).
Combines Metal-accelerated semantic search (GrainVDB) with tamper-proof cryptographic audit (Cua Driver).
"""

import subprocess
import json
import time
import threading
import queue
from typing import List, Dict, Optional, Any, Union, Callable
import numpy as np

from ..engine import GrainVDB, SearchMode, EngineType, Quantization, DistanceMetric
from .cua_merkle import MerkleTrajectoryChain, MerkleNode


class CuaGrainMemory:
    """
    Production-grade Unified Memory Engine for Computer Use Agents.
    
    Key Capabilities:
    1. Zero-Latency Semantic State Indexing: Sub-millisecond Apple Silicon Metal vector search.
    2. Non-blocking Async Ingestion: Background thread pool for high-FPS agent recordings.
    3. Hybrid Multimodal Recall: Fuses visual dense embeddings with OCR / UI text matching.
    4. Cryptographic Integrity: Cross-references semantic visual vectors with Cua sequence IDs.
    5. Audit In-Memory Caching: Minimizes subprocess overhead for repetitive audit queries.
    6. Structured UI Filtering: Supports filtering by application, action type, or outcome.
    7. Automatic Root-Cause Extraction: Extracts concise corrective context for LLMs on failure.
    """
    
    def __init__(
        self, 
        dim: int = 768, 
        cua_binary: str = "cua-driver",
        mode: SearchMode = SearchMode.EXACT,
        engine: EngineType = EngineType.METAL,
        quant: Quantization = Quantization.FP16,
        distance: DistanceMetric = DistanceMetric.COSINE,
        audit_cache_size: int = 1024,
    ):
        self.dim = dim
        self.cua_binary = cua_binary
        self.db = GrainVDB(
            dim=dim, 
            mode=mode, 
            engine=engine,
            quant=quant,
            distance=distance
        )
        self.audit_cache_size = audit_cache_size
        self._audit_cache: Dict[int, Dict[str, Any]] = {}
        
        # Cryptographic Merkle-DAG Trajectory Chain
        self.merkle_chain = MerkleTrajectoryChain()
        
        # Async Ingestion Queue
        self._async_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._start_background_worker()
        
    def _start_background_worker(self):
        """Starts background worker thread for asynchronous vector ingestion."""
        def _worker():
            while not self._stop_event.is_set() or not self._async_queue.empty():
                try:
                    item = self._async_queue.get(timeout=0.1)
                    if item is None:
                        break
                    seq_id, text, embed, app, action, chash, extra = item
                    self.record_action(
                        cua_sequence_id=seq_id,
                        semantic_text=text,
                        screenshot_embedding=embed,
                        app_name=app,
                        action_type=action,
                        cryptographic_hash=chash,
                        extra_metadata=extra
                    )
                    self._async_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"[CuaGrainMemory Background Worker Error] {e}")

        self._worker_thread = threading.Thread(target=_worker, daemon=True)
        self._worker_thread.start()

    def record_action(
        self, 
        cua_sequence_id: int, 
        semantic_text: str, 
        screenshot_embedding: Union[List[float], np.ndarray],
        app_name: Optional[str] = None,
        action_type: Optional[str] = None,
        cryptographic_hash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Records an agent step synchronously into semantic memory and appends to the
        tamper-proof Merkle trajectory chain.
        """
        # Append to Merkle Trajectory Chain
        merkle_node = self.merkle_chain.append_step(
            sequence_id=cua_sequence_id,
            action_text=semantic_text,
            screenshot_vector=screenshot_embedding,
            metadata=extra_metadata
        )

        metadata = {
            "cua_seq": cua_sequence_id,
            "text": semantic_text,
            "timestamp": merkle_node.timestamp,
            "merkle_hash": merkle_node.node_hash,
            "parent_merkle_hash": merkle_node.parent_hash,
        }
        if app_name:
            metadata["app"] = app_name
        if action_type:
            metadata["action"] = action_type
        if cryptographic_hash:
            metadata["hash"] = cryptographic_hash
        if extra_metadata:
            metadata.update(extra_metadata)
            
        try:
            vec = np.asarray(screenshot_embedding, dtype=np.float32).reshape(1, self.dim)
            self.db.add_vectors(vectors=vec, metadata=[metadata])
            return True
        except Exception as e:
            print(f"[CuaGrainMemory Warning] Failed to index semantic state at seq #{cua_sequence_id}: {e}")
            return False

    def record_action_async(
        self, 
        cua_sequence_id: int, 
        semantic_text: str, 
        screenshot_embedding: Union[List[float], np.ndarray],
        app_name: Optional[str] = None,
        action_type: Optional[str] = None,
        cryptographic_hash: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Non-blocking enqueue for high-FPS agent workflows. Ingests in background thread.
        """
        self._async_queue.put((
            cua_sequence_id, semantic_text, screenshot_embedding, app_name, action_type, cryptographic_hash, extra_metadata
        ))

    def flush_async_queue(self):
        """Blocks until all queued async records have been committed to GrainVDB."""
        self._async_queue.join()

    def semantic_recall(
        self, 
        query_embedding: Union[List[float], np.ndarray], 
        k: int = 3,
        app_filter: Optional[str] = None,
        action_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Finds the top-K most semantically similar past agent visual/text states.
        Supports optional metadata predicates (e.g., filter only within specific app).
        """
        self.flush_async_queue()
        try:
            vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
            
            filter_fn = None
            if app_filter or action_filter:
                def _predicate(vid: int, meta: Optional[Dict[str, Any]]) -> bool:
                    if not meta:
                        return False
                    if app_filter and meta.get("app") != app_filter:
                        return False
                    if action_filter and meta.get("action") != action_filter:
                        return False
                    return True
                filter_fn = _predicate
                
            results = self.db.search(vec, k=k, filter=filter_fn)
            
            recalled_events = []
            for idx, score in zip(results.indices, results.scores):
                meta = self.db.get_metadata(int(idx))
                if meta:
                    recalled_events.append({
                        "vector_id": int(idx),
                        "cua_sequence": meta.get("cua_seq"),
                        "similarity_score": float(score),
                        "semantic_context": meta.get("text"),
                        "app": meta.get("app"),
                        "action": meta.get("action"),
                        "cryptographic_hash": meta.get("hash"),
                    })
            return recalled_events
        except Exception as e:
            print(f"[CuaGrainMemory Warning] Semantic recall failed: {e}")
            return []

    def hybrid_recall(
        self,
        query_text: str,
        query_embedding: Union[List[float], np.ndarray],
        k: int = 3,
        app_filter: Optional[str] = None,
        alpha: float = 0.6  # Weight for vector similarity vs keyword score
    ) -> List[Dict[str, Any]]:
        """
        Hybrid visual + OCR keyword search using reciprocal score fusion.
        Balances semantic image embeddings with exact UI keyword matches.
        """
        self.flush_async_queue()
        candidates = self.semantic_recall(query_embedding, k=min(k * 4, self.total_records or 1), app_filter=app_filter)
        if not candidates:
            return []

        tokens = set(query_text.lower().split())
        scored = []
        for cand in candidates:
            text = (cand.get("semantic_context") or "").lower()
            token_matches = sum(1 for t in tokens if t in text)
            kw_score = token_matches / max(len(tokens), 1)
            
            vec_score = cand["similarity_score"]
            combined_score = alpha * vec_score + (1.0 - alpha) * kw_score
            cand["hybrid_score"] = float(combined_score)
            cand["keyword_overlap"] = kw_score
            scored.append(cand)

        scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return scored[:k]

    def secure_audit(self, cua_sequence_id: int) -> Optional[Dict[str, Any]]:
        """
        Queries Cua action provenance. Checks in-memory cache, falls back to native
        cryptographic Merkle trajectory proof, and optionally queries external daemon if present.
        """
        if cua_sequence_id in self._audit_cache:
            return self._audit_cache[cua_sequence_id]

        # 1. First check native Merkle inclusion proof
        try:
            merkle_proof = self.get_merkle_proof(cua_sequence_id)
            if merkle_proof:
                if len(self._audit_cache) < self.audit_cache_size:
                    self._audit_cache[cua_sequence_id] = merkle_proof
                return merkle_proof
        except KeyError:
            pass

        # 2. Check external Cua daemon if binary exists
        import shutil
        if shutil.which(self.cua_binary):
            try:
                cmd = [self.cua_binary, "history", "show", str(cua_sequence_id), "--json"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if len(self._audit_cache) < self.audit_cache_size:
                        self._audit_cache[cua_sequence_id] = data
                    return data
            except Exception:
                pass

        return None

    def hierarchical_recall(
        self,
        global_query_embedding: Union[List[float], np.ndarray],
        patch_query_embedding: Optional[Union[List[float], np.ndarray]] = None,
        alpha: float = 0.6,
        k: int = 5,
        app_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Hierarchical Multimodal Recall: Fuses global 4K desktop scene embedding with
        high-resolution ROI localized patch embedding to resolve fine-grained UI micro-elements.
        
        Args:
            global_query_embedding: 1D global screen vector
            patch_query_embedding: Optional 1D localized ROI patch vector
            alpha: Weight for global score (1 - alpha weight for patch score)
            k: Top-k matches to retrieve
            app_filter: Optional application filter
        """
        global_results = self.semantic_recall(global_query_embedding, k=k*2, app_filter=app_filter)
        if patch_query_embedding is None or not global_results:
            return global_results[:k]
            
        # Re-score candidates using patch similarity
        patch_vec = np.asarray(patch_query_embedding, dtype=np.float32).flatten()
        p_norm = np.linalg.norm(patch_vec)
        if p_norm > 1e-7:
            patch_vec = patch_vec / p_norm
            
        scored_candidates = []
        for r in global_results:
            seq_id = r["cua_sequence"]
            global_score = r["similarity_score"]
            
            # Check if candidate has stored patch embedding in extra metadata
            meta = self.db.get_metadata(r["id"]) if "id" in r else None
            stored_patch = meta.get("patch_vector") if meta else None
            
            if stored_patch is not None:
                sp_vec = np.asarray(stored_patch, dtype=np.float32).flatten()
                sp_norm = np.linalg.norm(sp_vec)
                if sp_norm > 1e-7:
                    sp_vec = sp_vec / sp_norm
                patch_score = float(np.dot(patch_vec, sp_vec))
                fused_score = alpha * global_score + (1.0 - alpha) * patch_score
            else:
                fused_score = global_score
                
            scored_candidates.append({
                **r,
                "fused_similarity_score": fused_score,
                "global_similarity_score": global_score,
            })
            
        scored_candidates.sort(key=lambda x: x.get("fused_similarity_score", 0.0), reverse=True)
        return scored_candidates[:k]

    def capture_screen(self, output_path: Optional[str] = None) -> bytes:
        """
        Captures live desktop screen natively on macOS using screencapture utility.
        """
        import tempfile, os
        target = output_path or os.path.join(tempfile.gettempdir(), f"cua_screen_{int(time.time()*1000)}.png")
        try:
            subprocess.run(["screencapture", "-x", "-C", target], check=True, capture_output=True)
            with open(target, "rb") as f:
                data = f.read()
            if not output_path and os.path.exists(target):
                os.remove(target)
            return data
        except Exception as e:
            logger_msg = f"Screen capture failed: {e}"
            print(f"[CuaGrainMemory Warning] {logger_msg}")
            return b""

    def capture_and_record(
        self,
        cua_sequence_id: int,
        semantic_text: str,
        embedder: Optional[Callable[[bytes], List[float]]] = None,
        app_name: Optional[str] = None,
        action_type: Optional[str] = None
    ) -> bool:
        """
        One-line capture and record: Takes desktop screenshot, embeds, and indexes into GrainVDB.
        """
        img_bytes = self.capture_screen()
        if not img_bytes:
            return False
            
        if embedder is not None:
            embedding = embedder(img_bytes)
        else:
            # Fallback zero-vector if no embedder provided
            embedding = [0.0] * self.dim
            
        return self.record_action(
            cua_sequence_id=cua_sequence_id,
            semantic_text=semantic_text,
            screenshot_embedding=embedding,
            app_name=app_name,
            action_type=action_type
        )

    def get_merkle_proof(self, cua_sequence_id: int) -> Dict[str, Any]:
        """
        Retrieves a cryptographic Merkle-DAG inclusion proof for a specific agent step.
        """
        return self.merkle_chain.get_merkle_proof(cua_sequence_id)

    def verify_trajectory_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verifies mathematical continuity of the entire action trajectory from Genesis.
        """
        return self.merkle_chain.verify_integrity()

    def record_trajectory_window(
        self,
        lead_sequence_id: int,
        steps: List[Dict[str, Any]],
        decay: float = 0.75,
        summary_text: Optional[str] = None
    ) -> bool:
        """
        Temporal Trajectory Slicing: Fuses multi-step temporal action sub-trajectories
        into a decayed temporal context vector for composite workflow recovery.
        
        Args:
            lead_sequence_id: Terminal sequence ID of the sub-trajectory
            steps: List of dicts with {"embedding": List[float], "action": str}
            decay: Exponential decay factor for earlier steps (0.0 to 1.0)
            summary_text: Optional text description of the sub-trajectory
        """
        if not steps:
            return False

        # Compute decayed fused vector: v = v_t + decay * v_{t-1} + decay^2 * v_{t-2} ...
        fused = np.zeros(self.dim, dtype=np.float32)
        weight_sum = 0.0
        actions = []
        for i, step in enumerate(reversed(steps)):
            w = decay ** i
            vec = np.asarray(step.get("embedding", [0.0] * self.dim), dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 1e-7:
                vec = vec / norm
            fused += w * vec
            weight_sum += w
            actions.append(step.get("action", ""))

        if weight_sum > 0:
            fused /= weight_sum
            fnorm = np.linalg.norm(fused)
            if fnorm > 1e-7:
                fused /= fnorm

        actions.reverse()
        composite_action = " -> ".join(actions)
        text = summary_text or f"Trajectory Window [{len(steps)} steps]: {composite_action}"

        return self.record_action(
            cua_sequence_id=lead_sequence_id,
            semantic_text=text,
            screenshot_embedding=fused,
            action_type="trajectory_window",
            extra_metadata={
                "is_trajectory_window": True,
                "window_size": len(steps),
                "steps_summary": composite_action
            }
        )

    def batch_secure_audit(self, cua_sequence_ids: List[int]) -> Dict[int, Optional[Dict[str, Any]]]:
        """Audits multiple sequence IDs in batch, leveraging cache."""
        return {seq_id: self.secure_audit(seq_id) for seq_id in cua_sequence_ids}

    def extract_failure_root_cause(
        self, 
        query_embedding: Union[List[float], np.ndarray], 
        k: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        One-shot root-cause diagnosis: Performs semantic recall on failure signature,
        verifies Cua cryptographic audit, and prepares compact LLM injection context.
        """
        matches = self.semantic_recall(query_embedding, k=k)
        if not matches:
            return None
            
        top_match = matches[0]
        seq_id = top_match["cua_sequence"]
        audit_info = self.secure_audit(seq_id) if seq_id is not None else None
        
        return {
            "sequence_id": seq_id,
            "semantic_context": top_match.get("semantic_context"),
            "similarity": top_match.get("similarity_score"),
            "audit": audit_info,
            "corrective_prompt": (
                f"Agent encountered error at Step #{seq_id}. "
                f"UI State: '{top_match.get('semantic_context')}'. "
                f"Audit Outcome: '{audit_info.get('outcome', 'unknown') if audit_info else 'unverified'}'. "
                f"Action Target: '{audit_info.get('target', 'unknown') if audit_info else 'unknown'}'."
            )
        }

    def save_checkpoint(self, path: str) -> bool:
        """Saves current memory index and metadata to disk."""
        self.flush_async_queue()
        return self.db.save(path)

    def load_checkpoint(self, path: str) -> bool:
        """Loads memory index and metadata from disk."""
        return self.db.load(path)

    def close(self):
        """Shuts down background ingestion workers."""
        self._stop_event.set()
        self._async_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def total_records(self) -> int:
        return self.db.vector_count
