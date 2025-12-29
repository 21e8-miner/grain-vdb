import ctypes
import numpy as np
import os
from typing import Tuple

class GrainVDB:
    """
    Python Bridge to the GrainVDB Native Metal Core.
    Build for Apple Silicon (M-series) with Unified Memory.
    No PyTorch/MPS dependencies. Purely native ctypes implementation.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.lib = None
        self.ctx = None
        self._load_native_library()
        
    def _load_native_library(self):
        # Locate the dylib relative to the library directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lib_name = "libgrainvdb.dylib"
        lib_path = os.path.join(root_dir, lib_name)
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"GrainVDB Native Binary {lib_name} not found at {lib_path}. Run ./build.sh first.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # Define C-API signatures
        # gv1_ctx_create(uint32_t rank, const char* library_path)
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        
        # gv1_data_feed(gv1_state_t* state, const float* buffer, uint32_t count)
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        
        # gv1_manifold_fold(...) -> returns total wall-time in ms (float)
        self.lib.gv1_manifold_fold.restype = ctypes.c_float
        self.lib.gv1_manifold_fold.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_uint32, 
            ctypes.POINTER(ctypes.c_uint64), 
            ctypes.POINTER(ctypes.c_float)
        ]
        
        # gv1_topology_audit(...) -> returns float (connectivity)
        self.lib.gv1_topology_audit.restype = ctypes.c_float
        self.lib.gv1_topology_audit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        
        # gv1_ctx_destroy(gv1_state_t* state)
        self.lib.gv1_ctx_destroy.argtypes = [ctypes.c_void_p]
        
        # Initialize context with relative metallib path
        metallib_path = os.path.join(root_dir, "grainvdb/gv_kernel.metallib")
        if not os.path.exists(metallib_path):
            raise FileNotFoundError(f"GrainVDB Metal Library not found at {metallib_path}. Ensure it exists in the grainvdb/ folder.")
            
        self.ctx = self.lib.gv1_ctx_create(self.dim, metallib_path.encode('utf-8'))
        
        if not self.ctx:
            raise RuntimeError("GrainVDB: Failed to initialize native context.")

    def add_vectors(self, vectors: np.ndarray):
        """
        Loads vectors into Unified Memory. 
        Input vectors are normalized internally to unit length for cosine similarity.
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.dim}, got {vectors.shape[1]}")
            
        # Implementation Detail: Internal normalization ensures valid cosine scores in search.
        v_f32 = vectors.astype(np.float32)
        norms = np.linalg.norm(v_f32, axis=1, keepdims=True)
        v_norm = v_f32 / (norms + 1e-9)
        
        data = np.ascontiguousarray(v_norm, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data))

    def query(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes native GPU similarity discovery + CPU Top-k selection.
        Returns: (indices, scores, total_latency_ms).
        """
        # Normalize probe vector
        q_f32 = query_vec.astype(np.float32)
        q_norm = q_f32 / (np.linalg.norm(q_f32) + 1e-9)
        
        probe = np.ascontiguousarray(q_norm, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        results_idx = np.zeros(k, dtype=np.uint64)
        results_scores = np.zeros(k, dtype=np.float32)
        
        idx_ptr = results_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        score_ptr = results_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Wall-time is measured end-to-end within the C++ layer.
        latency_ms = self.lib.gv1_manifold_fold(self.ctx, p_ptr, k, idx_ptr, score_ptr)
        
        return results_idx, results_scores, latency_ms

    def audit(self, indices: np.ndarray) -> float:
        """
        Calculates neighborhood connectivity density (Audit Heuristic).
        """
        idx_ptr = indices.astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.gv1_topology_audit(self.ctx, idx_ptr, len(indices))

    def __del__(self):
        if self.ctx and self.lib:
            self.lib.gv1_ctx_destroy(self.ctx)
