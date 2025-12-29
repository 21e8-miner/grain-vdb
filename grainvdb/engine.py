import ctypes
import numpy as np
import os
from typing import Tuple

class GrainVDB:
    """
    Python Bridge to the GrainVDB Native Metal Core.
    High-performance vector search for Apple Silicon via Unified Memory.
    No PyTorch dependencies. Purely native implementation.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.lib = None
        self.ctx = None
        self._load_library()
        
    def _load_library(self):
        # Resolve library path relative to this file
        root = os.path.dirname(os.path.abspath(__file__))
        lib_name = "libgrainvdb.dylib"
        lib_path = os.path.join(root, lib_name)
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Native binary {lib_name} not found at {lib_path}. Run ./build.sh first.")
            
        self.lib = ctypes.CDLL(lib_path)
        
        # Define API Signatures
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
        
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32]
        
        self.lib.gv1_manifold_fold.restype = ctypes.c_float
        self.lib.gv1_manifold_fold.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_float), 
            ctypes.c_uint32, 
            ctypes.POINTER(ctypes.c_uint64), 
            ctypes.POINTER(ctypes.c_float)
        ]
        
        self.lib.gv1_topology_audit.restype = ctypes.c_float
        self.lib.gv1_topology_audit.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.c_uint32]
        
        self.lib.gv1_ctx_destroy.argtypes = [ctypes.c_void_p]
        
        # Initialize Context with relative metallib path
        metallib = os.path.join(root, "gv_kernel.metallib")
        if not os.path.exists(metallib):
             raise FileNotFoundError(f"Metal kernel not found at {metallib}. Run ./build.sh first.")
            
        self.ctx = self.lib.gv1_ctx_create(self.dim, metallib.encode('utf-8'))
        if not self.ctx:
            raise RuntimeError(f"Native initialization failed using library: {metallib}")

    def add_vectors(self, vectors: np.ndarray, assume_normalized: bool = False):
        """
        Uploads vectors to Unified Memory.
        vectors: float32 array of shape (N, dim)
        assume_normalized: If True, skips internal normalization (faster if data is already unit length).
        """
        if vectors.shape[1] != self.dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.dim}")
            
        # Normalization (Requirement for Hardware-Optimized Cosine)
        if not assume_normalized:
            v_f32 = vectors.astype(np.float32)
            norms = np.linalg.norm(v_f32, axis=1, keepdims=True)
            v_norm = v_f32 / (norms + 1e-9)
            data = np.ascontiguousarray(v_norm, dtype=np.float32)
        else:
            data = np.ascontiguousarray(vectors, dtype=np.float32)

        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data))

    def query(self, query_vec: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes native GPU search + CPU top-f selection.
        Returns: (indices, scores, kernel_execution_ms).
        Note: kernel_execution_ms is the time spent in the C++ driver (dispatch + wait + select).
        """
        # Normalize probe
        q_norm = query_vec.astype(np.float32)
        q_norm = q_norm / (np.linalg.norm(q_norm) + 1e-9)
        
        probe = np.ascontiguousarray(q_norm, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        idx = np.zeros(k, dtype=np.uint64)
        scores = np.zeros(k, dtype=np.float32)
        
        idx_ptr = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        score_ptr = scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Time measured in C++ layer
        kernel_ms = self.lib.gv1_manifold_fold(self.ctx, p_ptr, k, idx_ptr, score_ptr)
        
        return idx, scores, kernel_ms

    def audit(self, indices: np.ndarray) -> float:
        """
        Calculates Semantic Consensus (Neighborhood Audit).
        Returns connectivity density among the retrieved results.
        """
        ptr = indices.astype(np.uint64).ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        return self.lib.gv1_topology_audit(self.ctx, ptr, len(indices))

    def __del__(self):
        if self.ctx and self.lib:
            self.lib.gv1_ctx_destroy(self.ctx)
