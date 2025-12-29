/**
 * @file gv_core.h
 * @brief GrainVDB Core API: Native Metal-Accelerated Vector Engine
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
 * @param rank Dimensionality of the vectors (must be a multiple of 4 for SIMD).
 * @param library_path Path to the compiled .metallib file.
 * @return Pointer to the allocated state, or NULL on failure.
 */
gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path);

/**
 * @brief Load float32 vectors into the unified memory manifold.
 * @param state Pointer to the VDB context.
 * @param buffer Flat float32 array of vectors.
 * @param count Number of vectors in the buffer.
 */
void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count);

/**
 * @brief Execute a brute-force similarity search on the manifold.
 * @return wall_latency_ms Total time (ms) including GPU dispatch and CPU Top-K
 * selection.
 */
float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag);

/**
 * @brief Calculate a neighborhood connectivity heuristic.
 * Measures integrated similarity density among results.
 */
float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count);

/**
 * @brief Clean up and release context resources.
 */
void gv1_ctx_destroy(gv1_state_t *state);

#ifdef __cplusplus
}
#endif

#endif // GV_CORE_H
