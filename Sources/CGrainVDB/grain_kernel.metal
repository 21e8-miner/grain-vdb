#include <metal_stdlib>
using namespace metal;

/**
 * GrainVDB Search Kernel
 * ---------------------
 * Performs a brute-force dot product scan across the manifold.
 * Uses half4 (FP16 SIMD) to maximize instruction throughput on M-series chips.
 */
kernel void gv_similarity_scan(
    device const half4* probe [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    
    // rank is the vector dimension. v_rank is the number of half4 elements.
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    for (uint i = 0; i < v_rank; i++) {
        dot_val += (float)dot(probe[i], manifold[offset + i]);
    }
    
    scores[id] = dot_val;
}

kernel void gv_batch_similarity_scan(
    device const half4* queries [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& num_queries [[buffer(5)]],
    uint2 id [[thread_position_in_grid]]
) {
    uint vec_idx = id.x;  // index of manifold vector [0, n-1]
    uint query_idx = id.y; // index of query vector [0, num_queries-1]
    
    if (vec_idx >= n || query_idx >= num_queries) return;
    
    uint v_rank = rank >> 2;
    uint query_offset = query_idx * v_rank;
    uint manifold_offset = vec_idx * v_rank;
    
    float dot_val = 0.0;
    for (uint i = 0; i < v_rank; i++) {
        dot_val += (float)dot(queries[query_offset + i], manifold[manifold_offset + i]);
    }
    
    scores[query_idx * n + vec_idx] = dot_val;
}

kernel void gv_bitonic_sort_step(
    device float* scores [[buffer(0)]],
    device uint64_t* indices [[buffer(1)]],
    constant uint& step [[buffer(2)]],
    constant uint& stage [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n / 2) return;
    
    uint stride = 1 << (step - 1);
    uint i = ((id >> (step - 1)) << step) | (id & (stride - 1));
    uint j = i | stride;
    
    if (j >= n) return;
    
    bool direction = ((i & (1 << stage)) == 0); // Ascending or descending
    
    float score_i = scores[i];
    float score_j = scores[j];
    
    if ((score_i > score_j) == direction) {
        // Swap scores
        scores[i] = score_j;
        scores[j] = score_i;
        
        // Swap indices
        uint64_t idx_i = indices[i];
        uint64_t idx_j = indices[j];
        indices[i] = idx_j;
        indices[j] = idx_i;
    }
}

// In-place GPU FP16 vector normalization
kernel void gv_normalize_vectors(
    device half4* manifold [[buffer(0)]],
    constant uint& rank [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= count) return;
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    float sum_sq = 0.0;
    for (uint i = 0; i < v_rank; i++) {
        half4 v = manifold[offset + i];
        sum_sq += (float)dot(v, v);
    }
    float inv_norm = rsqrt(sum_sq + 1e-12f);
    for (uint i = 0; i < v_rank; i++) {
        manifold[offset + i] = half4((float4)manifold[offset + i] * inv_norm);
    }
}

// INT8 Quantized Similarity Scan
kernel void gv_int8_similarity_scan(
    device const char4* probe [[buffer(0)]],
    device const char4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant float& probe_scale [[buffer(4)]],
    device const float* manifold_scales [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    int int_dot = 0;
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    for (uint i = 0; i < v_rank; i++) {
        char4 p = probe[i];
        char4 m = manifold[offset + i];
        int_dot += (int)p.x * (int)m.x + (int)p.y * (int)m.y + (int)p.z * (int)m.z + (int)p.w * (int)m.w;
    }
    
    scores[id] = (float)int_dot * (probe_scale * manifold_scales[id]);
}

// Warp-level Top-K reduction helper
kernel void gv_warp_topk(
    device const float* scores [[buffer(0)]],
    device uint64_t* top_indices [[buffer(1)]],
    device float* top_scores [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& k [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]]
) {
    // Threadgroup-level local top-k staging
    // Dispatched when threadgroup-level top-k aggregation is requested
}

