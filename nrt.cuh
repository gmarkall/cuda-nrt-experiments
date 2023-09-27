#pragma once

#include <cuda/atomic>

typedef void* (*NRT_malloc_func)(size_t size);
typedef void* (*NRT_realloc_func)(void* ptr, size_t new_size);
typedef void (*NRT_free_func)(void* ptr);
  
struct {
  bool enabled;
  cuda::atomic<size_t, cuda::thread_scope_device> alloc;
  cuda::atomic<size_t, cuda::thread_scope_device> free;
  cuda::atomic<size_t, cuda::thread_scope_device> mi_alloc;
  cuda::atomic<size_t, cuda::thread_scope_device> mi_free;
} NRT_Stats;

static __device__ NRT_Stats *memsys_stats;


__global__ void init_stats(NRT_Stats *stats, bool stats_enabled)
{
  memsys_stats = stats;
  memsys_stats->enabled = stats_enabled;
  memsys_stats->alloc = 0;
  memsys_stats->free = 0;
  memsys_stats->mi_alloc = 0;
  memsys_stats->mi_free = 0;
}
__device__ void* malloc_wrapper(size_t size)
{
  if (stats->enabled)
    stats->alloc++;

  return malloc(size);
}

__device__ void free_wrapper(void* ptr)
{
  if (stats->enabled)
    stats->free++;

  free(ptr);
}


