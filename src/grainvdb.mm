/**
 * @file grainvdb.mm
 * @brief Native Metal Driver for GrainVDB
 * Licensed under the MIT License.
 */

#include "gv_core.h"
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <vector>

typedef uint16_t gv_half_t;

// FP32 -> FP16 Conversion
static gv_half_t f32_to_f16(float f) {
  uint32_t i = *((uint32_t *)&f);
  int s = (i >> 16) & 0x00008000;
  int e = ((i >> 23) & 0x000000ff) - (127 - 15);
  int m = i & 0x007fffff;
  if (e <= 0) {
    if (e < -10)
      return s;
    m = (m | 0x00800000) >> (1 - e);
    return s | (m >> 13);
  } else if (e == 0xff - (127 - 15)) {
    return (m == 0) ? (s | 0x7c00) : (s | 0x7c00 | (m >> 13));
  } else {
    if (e > 30)
      return s | 0x7c00;
    return s | (e << 10) | (m >> 13);
  }
}

// FP16 -> FP32 Conversion
static float f16_to_f32(gv_half_t h) {
  uint32_t s = (h & 0x8000) << 16;
  uint32_t e = (h & 0x7c00) >> 10;
  uint32_t m = (h & 0x03ff) << 13;
  if (e == 0x1f) {
    e = 0xff;
  } else if (e == 0) {
    if (m != 0) {
      while (!(m & 0x00800000)) {
        m <<= 1;
        e--;
      }
      e++;
    }
  } else {
    e = e + (127 - 15);
  }
  uint32_t res = s | (e << 23) | m;
  return *((float *)&res);
}

struct gv1_state_t {
  id<MTLDevice> dev;
  id<MTLCommandQueue> q;
  id<MTLComputePipelineState> p_state;
  uint32_t rnk;
  id<MTLBuffer> m_buf; // Manifold Buffer (FP16)
  uint32_t count;
  id<MTLLibrary> lib;
};

gv1_state_t *gv1_ctx_create(uint32_t rank, const char *library_path) {
  gv1_state_t *state = new gv1_state_t();
  state->dev = MTLCreateSystemDefaultDevice();
  if (!state->dev)
    return nullptr;
  state->q = [state->dev newCommandQueue];
  state->rnk = rank;
  state->count = 0;
  state->m_buf = nil;

  NSError *err = nil;
  NSString *path = [NSString stringWithUTF8String:library_path];
  state->lib = [state->dev newLibraryWithFile:path error:&err];

  if (!state->lib) {
    std::cerr
        << "[GrainVDB Native Error] Failed to load Metal library from path: "
        << library_path << std::endl;
    delete state;
    return nullptr;
  }

  id<MTLFunction> fn = [state->lib newFunctionWithName:@"gv_similarity_scan"];
  if (!fn) {
    std::cerr << "[GrainVDB Native Error] Failed to find kernel "
                 "'gv_similarity_scan' in library."
              << std::endl;
    delete state;
    return nullptr;
  }

  state->p_state = [state->dev newComputePipelineStateWithFunction:fn
                                                             error:&err];
  if (!state->p_state) {
    std::cerr << "[GrainVDB Native Error] Pipeline state creation failed."
              << std::endl;
    delete state;
    return nullptr;
  }

  return state;
}

void gv1_data_feed(gv1_state_t *state, const float *buffer, uint32_t count) {
  uint32_t total_elements = count * state->rnk;
  size_t byte_len = total_elements * sizeof(gv_half_t);

  // Allocate Shared Unified Memory
  state->m_buf = [state->dev newBufferWithLength:byte_len
                                         options:MTLResourceStorageModeShared];
  gv_half_t *dst = (gv_half_t *)[state->m_buf contents];

  // Ingest data with FP32->FP16 conversion
  for (uint32_t i = 0; i < total_elements; i++) {
    dst[i] = f32_to_f16(buffer[i]);
  }
  state->count = count;
}

float gv1_manifold_fold(gv1_state_t *state, const float *probe, uint32_t top,
                        uint64_t *result_map, float *result_mag) {
  if (state->count == 0 || !state->m_buf)
    return 0.0f;

  // 1. Setup Query Buffer
  id<MTLBuffer> p_buf =
      [state->dev newBufferWithLength:state->rnk * sizeof(gv_half_t)
                              options:MTLResourceStorageModeShared];
  gv_half_t *qp = (gv_half_t *)[p_buf contents];
  for (uint32_t i = 0; i < state->rnk; i++)
    qp[i] = f32_to_f16(probe[i]);

  // 2. Setup Score Buffer
  id<MTLBuffer> s_buf =
      [state->dev newBufferWithLength:state->count * sizeof(float)
                              options:MTLResourceStorageModeShared];

  // START TIMER: Includes GPU latency and CPU top-k selection overhead
  auto t_start = std::chrono::high_resolution_clock::now();

  // 3. Dispatch GPU Compute
  id<MTLCommandBuffer> c_buf = [state->q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [c_buf computeCommandEncoder];
  [enc setComputePipelineState:state->p_state];
  [enc setBuffer:p_buf offset:0 atIndex:0];
  [enc setBuffer:state->m_buf offset:0 atIndex:1];
  [enc setBuffer:s_buf offset:0 atIndex:2];
  [enc setBytes:&state->rnk length:sizeof(uint32_t) atIndex:3];

  MTLSize g_sz = MTLSizeMake(state->count, 1, 1);
  NSUInteger max_t = state->p_state.maxTotalThreadsPerThreadgroup;
  MTLSize t_sz = MTLSizeMake(std::min((uint32_t)max_t, state->count), 1, 1);

  [enc dispatchThreads:g_sz threadsPerThreadgroup:t_sz];
  [enc endEncoding];

  [c_buf commit];
  [c_buf waitUntilCompleted];

  // 4. CPU Top-k Selection (Priority Queue) over the Shared Score Buffer
  float *scores = (float *)[s_buf contents];
  typedef std::pair<float, uint64_t> Res;
  std::priority_queue<Res, std::vector<Res>, std::greater<Res>> pq;

  for (uint64_t i = 0; i < state->count; i++) {
    if (pq.size() < top) {
      pq.push({scores[i], i});
    } else if (scores[i] > pq.top().first) {
      pq.pop();
      pq.push({scores[i], i});
    }
  }

  // Capture results in descending order
  for (int i = (int)top - 1; i >= 0; i--) {
    result_mag[i] = pq.top().first;
    result_map[i] = pq.top().second;
    pq.pop();
  }

  auto t_end = std::chrono::high_resolution_clock::now();
  // Return explicit total milliseconds
  return (float)(std::chrono::duration_cast<std::chrono::microseconds>(t_end -
                                                                       t_start)
                     .count()) /
         1000.0f;
}

float gv1_topology_audit(gv1_state_t *state, const uint64_t *map,
                         uint32_t count) {
  if (count < 2 || !state->m_buf)
    return 1.0f;

  gv_half_t *m_ptr = (gv_half_t *)[state->m_buf contents];
  std::vector<std::vector<float>> neighbors(count,
                                            std::vector<float>(state->rnk));

  for (uint32_t i = 0; i < count; i++) {
    uint64_t offset = map[i] * state->rnk;
    for (uint32_t k = 0; k < state->rnk; k++) {
      neighbors[i][k] = f16_to_f32(m_ptr[offset + k]);
    }
  }

  int connections = 0;
  const float threshold = 0.85f;
  for (uint32_t i = 0; i < count; i++) {
    for (uint32_t j = i + 1; j < count; j++) {
      float dot = 0.0f;
      for (uint32_t k = 0; k < state->rnk; k++)
        dot += neighbors[i][k] * neighbors[j][k];
      if (dot > threshold)
        connections++;
    }
  }

  int total_pairs = (count * (count - 1)) / 2;
  return (float)connections / (float)total_pairs;
}

void gv1_ctx_destroy(gv1_state_t *state) {
  if (state)
    delete state;
}
