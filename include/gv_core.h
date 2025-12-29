/**
 * GrainVDB Core API
 * -----------------
 * Native Metal-Accelerated Vector Engine for Apple Silicon.
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
 * @param rank The dimensionality of the vectors.
 * @param library_path Path to the gv_kernel.metallib.
 */
gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path);

/**
 * @brief Load vectors into the Unified Memory buffer.
 * @param count Number of vectors to load.
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count);

/**
 * @brief Perform a brute-force query on the GPU.
 * @param top Number of results to return.
 * @param result_map Array to store result indices (uint64_t).
 * @param result_mag Array to store result scores (float).
 * @return kernel_latency_ms Measured GPU wall-time in milliseconds.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Calculate neighborhood connectivity (Audit).
 * @return connectivity_score 0.0 to 1.0 (Higher is more consistent).
 */
float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count);

/**
 * @brief Destroy the context and free resources.
 */
void gv1_ctx_destroy(gv1_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // GV_CORE_H
