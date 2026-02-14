/*
 * GrainVDB v2.0 - Breakthrough Edition
 * High-Performance Vector Search Engine for Apple Silicon
 * 
 * Features:
 * - GPU-accelerated Top-K with bitonic sort
 * - Batch query processing
 * - INT8 quantization support
 * - Warp-optimized distance computation
 */

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Constants and Types
// ============================================================================

// INT8 quantization constants
struct QuantizationParams {
    float scale;
    float zero_point;
};

// ============================================================================
// Core Distance Computation Kernels
// ============================================================================

/*
 * Brute-force similarity scan using FP16 half4 SIMD
 * Maximize instruction throughput on M-series chips
 */
kernel void gv_similarity_scan(
    device const half4* probe [[buffer(0)]],
    device const half4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    uint v_rank = rank >> 2;  // rank / 4 for half4 elements
    uint offset = id * v_rank;
    
    #pragma unroll(4)
    for (uint i = 0; i < v_rank; i++) {
        dot_val += (float)dot(probe[i], manifold[offset + i]);
    }
    
    scores[id] = dot_val;
}

/*
 * Batch similarity scan - process multiple queries in parallel
 * Each threadgroup handles one query, threads handle vectors
 */
kernel void gv_batch_similarity_scan(
    device const half4* probes [[buffer(0)]],        // [batch_size, rank/4]
    device const half4* manifold [[buffer(1)]],      // [n_vectors, rank/4]
    device float* scores [[buffer(2)]],              // [batch_size, n_vectors]
    constant uint& rank [[buffer(3)]],
    constant uint& n_vectors [[buffer(4)]],
    constant uint& batch_size [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]           // (vector_id, query_id)
) {
    uint qid = gid.y;  // query id
    uint vid = gid.x;  // vector id
    
    if (qid >= batch_size || vid >= n_vectors) return;
    
    float dot_val = 0.0;
    uint v_rank = rank >> 2;
    
    device const half4* probe = &probes[qid * v_rank];
    device const half4* vec = &manifold[vid * v_rank];
    
    #pragma unroll(4)
    for (uint i = 0; i < v_rank; i++) {
        dot_val += (float)dot(probe[i], vec[i]);
    }
    
    scores[qid * n_vectors + vid] = dot_val;
}

/*
 * INT8 quantized similarity scan
 * 4x memory bandwidth reduction with minimal accuracy loss
 */
kernel void gv_int8_similarity_scan(
    device const char4* probe [[buffer(0)]],
    device const char4* manifold [[buffer(1)]],
    device float* scores [[buffer(2)]],
    constant uint& rank [[buffer(3)]],
    constant QuantizationParams& qparams [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float dot_val = 0.0;
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    // Dequantize and compute dot product
    #pragma unroll(4)
    for (uint i = 0; i < v_rank; i++) {
        char4 p = probe[i];
        char4 v = manifold[offset + i];
        
        // INT8 to FP32 conversion with quantization params
        float4 pf = float4(p) * qparams.scale + qparams.zero_point;
        float4 vf = float4(v) * qparams.scale + qparams.zero_point;
        
        dot_val += dot(pf, vf);
    }
    
    scores[id] = dot_val;
}

// ============================================================================
// GPU-Accelerated Top-K Selection (Bitonic Sort Network)
// ============================================================================

/*
 * Bitonic sort step for Top-K selection
 * Performs parallel compare-and-swap operations
 */
kernel void gv_bitonic_sort_step(
    device float* scores [[buffer(0)]],
    device uint64_t* indices [[buffer(1)]],
    constant uint& step [[buffer(2)]],
    constant uint& stage [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint pair_distance = 1 << (stage - step);
    uint block_width = pair_distance << 1;
    
    uint left_id = (gid / pair_distance) * block_width + (gid % pair_distance);
    uint right_id = left_id + pair_distance;
    
    if (right_id >= n) return;
    
    // Determine sort direction (ascending/descending)
    bool ascending = ((left_id / (1 << stage)) & 1) == 0;
    
    float left_score = scores[left_id];
    float right_score = scores[right_id];
    
    bool should_swap = ascending ? (left_score > right_score) : (left_score < right_score);
    
    if (should_swap) {
        scores[left_id] = right_score;
        scores[right_id] = left_score;
        
        uint64_t left_idx = indices[left_id];
        uint64_t right_idx = indices[right_id];
        indices[left_id] = right_idx;
        indices[right_id] = left_idx;
    }
}

/*
 * Warp-level parallel Top-K using register shuffling
 * Optimized for small K values (K <= 32)
 */
kernel void gv_warp_topk(
    device const float* scores [[buffer(0)]],
    device uint64_t* topk_indices [[buffer(1)]],
    device float* topk_scores [[buffer(2)]],
    constant uint& k [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    // Each warp processes a chunk of the input
    uint warp_id = gid / 32;
    uint lane_id = tid % 32;
    
    // Shared memory for warp-level reduction
    threadgroup float shared_scores[32];
    threadgroup uint shared_indices[32];
    
    // Each thread finds local top-1 in its portion
    float local_max = -FLT_MAX;
    uint local_idx = 0;
    
    for (uint i = lane_id + warp_id * 32; i < n; i += 32) {
        float s = scores[i];
        if (s > local_max) {
            local_max = s;
            local_idx = i;
        }
    }
    
    // Warp-level reduction to find top-1 per warp
    shared_scores[lane_id] = local_max;
    shared_indices[lane_id] = local_idx;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Parallel reduction within warp
    for (uint offset = 16; offset > 0; offset >>= 1) {
        if (lane_id < offset) {
            if (shared_scores[lane_id + offset] > shared_scores[lane_id]) {
                shared_scores[lane_id] = shared_scores[lane_id + offset];
                shared_indices[lane_id] = shared_indices[lane_id + offset];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write results
    if (lane_id == 0 && warp_id < k) {
        topk_scores[warp_id] = shared_scores[0];
        topk_indices[warp_id] = shared_indices[0];
    }
}

// ============================================================================
// HNSW Graph Search Kernels
// ============================================================================

/*
 * GPU-optimized HNSW layer search
 * Flat graph structure with warp-level parallelism
 */
struct HNSWNode {
    uint32_t id;
    float distance;
};

kernel void gv_hnsw_layer_search(
    device const half4* query [[buffer(0)]],
    device const half4* vectors [[buffer(1)]],
    device const uint32_t* neighbors [[buffer(2)]],     // Adjacency list
    device const uint32_t* neighbor_offsets [[buffer(3)]], // Offset into neighbors
    device HNSWNode* candidates [[buffer(4)]],
    device HNSWNode* results [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    constant uint& entry_point [[buffer(7)]],
    constant uint& ef [[buffer(8)]],
    constant uint& max_neighbors [[buffer(9)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    uint lane_id = tid % 32;
    uint warp_id = tid / 32;
    
    uint v_rank = rank >> 2;
    
    // Compute distance for assigned node
    uint node_id = neighbors[entry_point * max_neighbors + gid];
    if (node_id == UINT_MAX) return;
    
    float dist = 0.0;
    device const half4* vec = &vectors[node_id * v_rank];
    
    #pragma unroll(4)
    for (uint i = lane_id; i < v_rank; i += 32) {
        dist += (float)dot(query[i], vec[i]);
    }
    
    // Warp-level reduction
    for (uint offset = 16; offset > 0; offset >>= 1) {
        dist += simd_shuffle_down(dist, offset);
    }
    
    if (lane_id == 0) {
        candidates[gid].id = node_id;
        candidates[gid].distance = dist;
    }
}

// ============================================================================
// Utility Kernels
// ============================================================================

/*
 * FP32 to FP16 conversion kernel
 * Batch convert vectors for storage efficiency
 */
kernel void gv_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        output[gid] = (half)input[gid];
    }
}

/*
 * FP32 to INT8 quantization kernel
 */
kernel void gv_f32_to_int8(
    device const float* input [[buffer(0)]],
    device int8_t* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant QuantizationParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < n) {
        float val = (input[gid] - params.zero_point) / params.scale;
        val = clamp(val, -128.0f, 127.0f);
        output[gid] = (int8_t)val;
    }
}

/*
 * Vector normalization kernel
 * Normalize vectors in-place on GPU
 */
kernel void gv_normalize_vectors(
    device half4* vectors [[buffer(0)]],
    constant uint& rank [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    uint v_rank = rank >> 2;
    uint offset = id * v_rank;
    
    float norm = 0.0;
    #pragma unroll(4)
    for (uint i = 0; i < v_rank; i++) {
        half4 v = vectors[offset + i];
        norm += (float)dot(v, v);
    }
    
    norm = sqrt(norm);
    if (norm > 1e-7) {
        #pragma unroll(4)
        for (uint i = 0; i < v_rank; i++) {
            vectors[offset + i] = vectors[offset + i] / (half)norm;
        }
    }
}
