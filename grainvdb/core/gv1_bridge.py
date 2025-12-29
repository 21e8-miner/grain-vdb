import ctypes
import numpy as np
import os

class GV1Engine:
    """
    Python Bridge to the Proprietary GV1 Manifold Engine (Real Tech Binary).
    """
    def __init__(self, rank=128):
        self.rank = rank
        self.ctx = None
        
        # Determine the logical path to the dylib
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lib_path = os.path.join(base_dir, "dist/libgrainvdb.dylib")
        
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"GrainVDB Native Core not found at {lib_path}. Run scripts/build_native.sh first.")
        
        self._load_lib(lib_path)

    def _load_lib(self, dylib_path):
        self.lib = ctypes.CDLL(dylib_path)
        
        self.lib.gv1_ctx_create.restype = ctypes.c_void_p
        self.lib.gv1_ctx_create.argtypes = [ctypes.c_uint32]
        
        self.lib.gv1_data_feed.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint32, ctypes.c_bool]
        
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
        
        self.ctx = self.lib.gv1_ctx_create(self.rank)

    def feed(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.lib.gv1_data_feed(self.ctx, ptr, len(data), False)

    def fold(self, probe, top=5):
        probe = np.ascontiguousarray(probe, dtype=np.float32)
        p_ptr = probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        res_maps = np.zeros(top, dtype=np.uint64)
        res_mags = np.zeros(top, dtype=np.float32)
        
        m_ptr = res_maps.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))
        mag_ptr = res_mags.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        latency = self.lib.gv1_manifold_fold(self.ctx, p_ptr, top, m_ptr, mag_ptr)
        return res_maps, res_mags, latency
