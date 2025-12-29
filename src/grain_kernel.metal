#include <metal_stdlib>
using namespace metal;

/**
 * GrainVDB Search Kernel
 * ---------------------
 * Vectorized dot product using FP16 (half4) for 4x instruction throughput.
 */
kernel void gv_similarity_scan(
    device const half4* probe [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    
    // Rank is dimensionality. v_rank is rank/4 for half4 SIMD.
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    for (uint i = 0; i < v_rank; i++) {
        dot_val += (float)dot(probe[i], manifold[offset + i]);
    }
    
    scores[id] = dot_val;
}
