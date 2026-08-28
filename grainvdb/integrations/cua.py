"""
Cua-Grain Memory Integration
Unified memory layer for Computer Use Agents using GrainVDB and Cua Driver.
"""

import subprocess
import json
from typing import List, Dict, Optional, Any
import numpy as np

from ..engine import GrainVDB, SearchMode, EngineType, Quantization, DistanceMetric

class CuaGrainMemory:
    """
    A unified memory layer for Computer Use Agents.
    Combines GrainVDB (semantic state) with Cua Driver (cryptographic audit).
    """
    def __init__(
        self, 
        dim: int = 768, 
        cua_binary: str = "cua-driver",
        mode: SearchMode = SearchMode.EXACT,
        engine: EngineType = EngineType.METAL,
        quant: Quantization = Quantization.FP16,
        distance: DistanceMetric = DistanceMetric.COSINE
    ):
        # Initialize GrainVDB with Metal GPU acceleration for batch ops by default
        self.db = GrainVDB(
            dim=dim, 
            mode=mode, 
            engine=engine,
            quant=quant,
            distance=distance
        )
        self.cua_binary = cua_binary
        
    def record_action(self, cua_sequence_id: int, semantic_text: str, screenshot_embedding: List[float]):
        """
        Called after every agent step. Stores the visual/text state in GrainVDB.
        Cua Driver automatically handles the secure audit log in the background.
        """
        metadata = {"cua_seq": cua_sequence_id, "text": semantic_text}
        
        try:
            # Zero-copy insert into GrainVDB
            self.db.add_vectors(
                vectors=np.array([screenshot_embedding], dtype=np.float32), 
                metadata=[metadata]
            )
        except Exception as e:
            # Fail gracefully - agent shouldn't stop if memory ingestion fails
            print(f"[Memory Warning] Failed to index semantic state: {e}")

    def semantic_recall(self, query_embedding: List[float], k: int = 3) -> List[Dict[str, Any]]:
        """
        Searches the agent's visual history instantly. 
        Returns the top K most similar past states and their Cua Sequence IDs.
        """
        try:
            results = self.db.search(np.array(query_embedding, dtype=np.float32), k=k)
            
            recalled_events = []
            for idx, score in zip(results.indices, results.scores):
                meta = self.db.get_metadata(idx)
                if meta:
                    recalled_events.append({
                        "cua_sequence": meta.get("cua_seq"),
                        "similarity_score": float(score),
                        "semantic_context": meta.get("text")
                    })
            return recalled_events
        except Exception as e:
            print(f"[Memory Warning] Semantic recall failed: {e}")
            return []

    def secure_audit(self, cua_sequence_id: int) -> Optional[Dict[str, Any]]:
        """
        Queries the Cua Driver encrypted history to verify exactly what 
        capabilities and actions were invoked at a specific sequence.
        """
        try:
            # Execute Cua CLI to pull the secure, encrypted audit log
            cmd = [self.cua_binary, "history", "show", str(cua_sequence_id), "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            return None
        except subprocess.CalledProcessError as e:
            print(f"[Audit Error] Cua Driver query failed: {e.stderr if e.stderr else e}")
            return None
        except json.JSONDecodeError:
            print("[Audit Error] Invalid JSON response from Cua.")
            return None
        except Exception as e:
            print(f"[Audit Error] Secure audit execution failed: {e}")
            return None
