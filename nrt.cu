#include "nrt.cuh"

extern "C"
__device__ int init_memsys(void* dummy_return, uint64_t memsys_ptr, bool stats_enabled)
{
  TheMSys = reinterpret_cast<NRT_MemSys*>(memsys_ptr);

  TheMSys->allocator.malloc = static_cast<NRT_malloc_func>(malloc_wrapper);
  TheMSys->allocator.realloc = static_cast<NRT_realloc_func>(nullptr);
  TheMSys->allocator.free = static_cast<NRT_free_func>(free_wrapper);

  TheMSys->stats.enabled = stats_enabled;
  TheMSys->stats.alloc = 0;
  TheMSys->stats.free = 0;
  TheMSys->stats.mi_alloc = 0;
  TheMSys->stats.mi_free = 0;

  return 0;
}

extern "C"
__device__ int sizeof_memsys(size_t *size)
{
  *size = sizeof(NRT_MemSys);
  return 0;
}

__global__ void init_stats(NRT_Stats *stats, bool stats_enabled)
{
  memsys_stats->enabled = stats_enabled;
  memsys_stats->alloc = 0;
  memsys_stats->free = 0;
  memsys_stats->mi_alloc = 0;
  memsys_stats->mi_free = 0;
  *stats = &memsys_stats;
}
