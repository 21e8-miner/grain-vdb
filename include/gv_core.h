/**
 * @file gv_core.h
 * @brief GrainVDB Core API: Native Metal Engine
 * Licensed under the MIT License.
 */

#ifndef GV_CORE_H
#define GV_CORE_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct gv1_state_t gv1_state_t;

/**
 * @brief Initialize the GrainVDB context.
 * @param rank Vector dimensionality (must be multiple of 4 for SIMD).
 * @param library_path Path to the compiled .metallib file.
 */
gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path);

/**
 * @brief Load vectors into the unified memory buffer.
 * @param buffer Float32 data.
 * @param count Number of vectors.
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count);

/**
 * @brief Perform brute-force similarity search.
 * @return total_latency_ms Wall-time including GPU execution and CPU Top-k
 * selection.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Connectivity heuristic for neighborhood consistency.
 */
float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count);

/**
 * @brief Destroy context and release memory.
 */
void gv1_ctx_destroy(gv1_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // GV_CORE_H
