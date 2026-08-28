/*
 * GrainVDB v2.0 - Breakthrough Edition
 * Core API Header
 * 
 * Copyright (c) 2025 - Present
 * Licensed under MIT License
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Version Information
// ============================================================================

#define GRAINVDB_VERSION_MAJOR 2
#define GRAINVDB_VERSION_MINOR 0
#define GRAINVDB_VERSION_PATCH 0

// ============================================================================
// Type Definitions
// ============================================================================

/* Opaque context handle */
typedef struct gv2_context gv2_context_t;

/* Search result structure */
typedef struct {
    uint64_t* indices;      // Top-K indices
    float* scores;          // Similarity scores
    float latency_ms;       // Query latency
    uint32_t num_results;   // Number of results returned
} gv2_search_result_t;

/* Quantization type */
typedef enum {
    GV2_QUANT_FP32 = 0,     // Full precision (32-bit)
    GV2_QUANT_FP16 = 1,     // Half precision (16-bit) - default
    GV2_QUANT_INT8 = 2,     // INT8 quantization (8-bit)
    GV2_QUANT_BF16 = 3,     // BFloat16 (16-bit, more range)
} gv2_quantization_t;

/* Search mode */
typedef enum {
    GV2_SEARCH_EXACT = 0,   // Exact brute-force search
    GV2_SEARCH_HNSW = 1,    // HNSW approximate search
    GV2_SEARCH_HYBRID = 2,  // Hybrid: HNSW + exact refinement
} gv2_search_mode_t;

/* Distance metric */
typedef enum {
    GV2_DIST_COSINE = 0,    // Cosine similarity (default)
    GV2_DIST_EUCLIDEAN = 1, // Euclidean distance (L2)
    GV2_DIST_DOT = 2,       // Raw dot product
    GV2_DIST_MANHATTAN = 3, // Manhattan distance (L1)
} gv2_distance_t;

/* HNSW configuration */
typedef struct {
    uint32_t M;                 // Max connections per node (default: 16)
    uint32_t ef_construction;   // Construction candidate pool (default: 200)
    uint32_t ef_search;         // Search candidate pool (default: 50)
    uint32_t max_elements;      // Maximum elements (default: 0 = unlimited)
} gv2_hnsw_config_t;

/* Execution engine type */
typedef enum {
    GV2_ENGINE_AUTO = 0,        // Automatic: CPU Accelerate for single/small N, Metal GPU for batch/large N
    GV2_ENGINE_ACCELERATE = 1,  // Apple Accelerate CPU (vDSP / AMX / NEON)
    GV2_ENGINE_METAL = 2,       // Metal GPU compute shaders
} gv2_engine_t;

/* Context configuration */
typedef struct {
    uint32_t dimension;         // Vector dimension (must be multiple of 4)
    gv2_quantization_t quant;   // Quantization mode
    gv2_distance_t distance;    // Distance metric
    gv2_search_mode_t mode;     // Search mode
    gv2_engine_t engine;        // Execution engine backend
    gv2_hnsw_config_t hnsw;     // HNSW configuration (if applicable)
    const char* metallib_path;  // Path to Metal library
    bool use_gpu_topk;          // Use GPU-accelerated Top-K (default: true)
    bool use_batch_processing;  // Enable batch query processing (default: true)
    uint32_t batch_size;        // Default batch size (default: 32)
} gv2_config_t;

/* Topology audit result */
typedef struct {
    float connectivity;         // Neighborhood connectivity [0, 1]
    float coherence;            // Semantic coherence score
    float entropy;              // Shannon entropy of result distribution
    uint32_t num_connections;   // Number of connected pairs
} gv2_audit_result_t;

/* Performance metrics */
typedef struct {
    float avg_latency_ms;       // Average query latency
    float p50_latency_ms;       // P50 latency
    float p95_latency_ms;       // P95 latency
    float p99_latency_ms;       // P99 latency
    float throughput_qps;       // Queries per second
    uint64_t total_queries;     // Total queries processed
    float gpu_utilization;      // GPU utilization percentage
    float memory_usage_mb;      // Memory usage in MB
} gv2_metrics_t;

// ============================================================================
// Default Configurations
// ============================================================================

static inline gv2_config_t gv2_default_config(void) {
    gv2_config_t config = {
        .dimension = 128,
        .quant = GV2_QUANT_FP16,
        .distance = GV2_DIST_COSINE,
        .mode = GV2_SEARCH_EXACT,
        .engine = GV2_ENGINE_AUTO,
        .hnsw = {
            .M = 16,
            .ef_construction = 200,
            .ef_search = 50,
            .max_elements = 0
        },
        .metallib_path = NULL,
        .use_gpu_topk = true,
        .use_batch_processing = true,
        .batch_size = 32
    };
    return config;
}

// ============================================================================
// Context Lifecycle
// ============================================================================

/* Create a new GrainVDB context */
gv2_context_t* gv2_ctx_create(const gv2_config_t* config);

/* Destroy context and free all resources */
void gv2_ctx_destroy(gv2_context_t* ctx);

/* Get last error message */
const char* gv2_get_error(gv2_context_t* ctx);

/* Clear error state */
void gv2_clear_error(gv2_context_t* ctx);

// ============================================================================
// Data Management
// ============================================================================

/* Add vectors to the database */
bool gv2_add_vectors(gv2_context_t* ctx, 
                     const float* vectors, 
                     uint32_t count,
                     const uint64_t* ids);

/* Remove vectors by ID */
bool gv2_remove_vectors(gv2_context_t* ctx, 
                        const uint64_t* ids, 
                        uint32_t count);

/* Get number of stored vectors */
uint32_t gv2_vector_count(gv2_context_t* ctx);

/* Get vector by ID (returns copy) */
bool gv2_get_vector(gv2_context_t* ctx, 
                    uint64_t id, 
                    float* output);

/* Update existing vector */
bool gv2_update_vector(gv2_context_t* ctx, 
                       uint64_t id, 
                       const float* vector);

/* Clear all vectors */
void gv2_clear(gv2_context_t* ctx);

// ============================================================================
// Search Operations
// ============================================================================

/* Single query search */
gv2_search_result_t* gv2_search(gv2_context_t* ctx,
                                const float* query,
                                uint32_t k);

/* Batch query search - BREAKTHROUGH: High throughput */
gv2_search_result_t** gv2_search_batch(gv2_context_t* ctx,
                                       const float* queries,
                                       uint32_t num_queries,
                                       uint32_t k);

/* Search with filter predicate */
typedef bool (*gv2_filter_fn)(uint64_t id, void* userdata);

gv2_search_result_t* gv2_search_filtered(gv2_context_t* ctx,
                                         const float* query,
                                         uint32_t k,
                                         gv2_filter_fn filter,
                                         void* userdata);

/* Free search result */
void gv2_free_result(gv2_search_result_t* result);

/* Free batch results */
void gv2_free_batch_results(gv2_search_result_t** results, uint32_t count);

// ============================================================================
// HNSW Index Operations
// ============================================================================

/* Build HNSW index (required before approximate search) */
bool gv2_hnsw_build(gv2_context_t* ctx);

/* Get HNSW index statistics */
typedef struct {
    uint32_t num_nodes;
    uint32_t num_edges;
    uint32_t max_level;
    float avg_degree;
    size_t memory_usage_bytes;
} gv2_hnsw_stats_t;

bool gv2_hnsw_get_stats(gv2_context_t* ctx, gv2_hnsw_stats_t* stats);

// ============================================================================
// Topology Audit (Semantic Coherence Detection)
// ============================================================================

/* Audit search results for semantic coherence */
gv2_audit_result_t gv2_audit(gv2_context_t* ctx,
                             const uint64_t* result_ids,
                             uint32_t count);

/* Batch audit multiple result sets */
gv2_audit_result_t* gv2_audit_batch(gv2_context_t* ctx,
                                    gv2_search_result_t** results,
                                    uint32_t num_results);

// ============================================================================
// Persistence
// ============================================================================

/* Save index to disk */
bool gv2_save(gv2_context_t* ctx, const char* path);

/* Load index from disk */
bool gv2_load(gv2_context_t* ctx, const char* path);

/* Memory-map index for zero-copy loading */
bool gv2_mmap(gv2_context_t* ctx, const char* path);

/* Get index file size estimate */
size_t gv2_estimate_size(gv2_context_t* ctx);

// ============================================================================
// Performance & Monitoring
// ============================================================================

/* Get performance metrics */
bool gv2_get_metrics(gv2_context_t* ctx, gv2_metrics_t* metrics);

/* Reset performance metrics */
void gv2_reset_metrics(gv2_context_t* ctx);

/* Set search parameters at runtime */
bool gv2_set_ef_search(gv2_context_t* ctx, uint32_t ef);
bool gv2_set_batch_size(gv2_context_t* ctx, uint32_t size);
bool gv2_set_engine(gv2_context_t* ctx, gv2_engine_t engine);
gv2_engine_t gv2_get_engine(gv2_context_t* ctx);

/* Warmup GPU (pre-compile pipelines) */
void gv2_warmup(gv2_context_t* ctx);

/* Synchronize GPU (wait for all operations) */
void gv2_synchronize(gv2_context_t* ctx);

// ============================================================================
// Advanced Features
// ============================================================================

/* Quantize existing vectors to lower precision */
bool gv2_quantize(gv2_context_t* ctx, gv2_quantization_t target_quant);

/* Get recommended quantization for dataset */
gv2_quantization_t gv2_recommend_quantization(uint32_t dimension, 
                                               uint32_t num_vectors);

/* Estimate recall@k for current configuration */
float gv2_estimate_recall(gv2_context_t* ctx, uint32_t k);
float gv2_estimate_reall(gv2_context_t* ctx, uint32_t k);

/* Set distance threshold for early termination */
void gv2_set_distance_threshold(gv2_context_t* ctx, float threshold);

#ifdef __cplusplus
}
#endif
