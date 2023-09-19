#include "nrt.cuh"

__device__ NRT_MemSys *TheMSys;

__device__ void* malloc_wrapper(size_t size)
{ 
  if (TheMSys->stats.enabled)
    TheMSys->stats.alloc++;

  return malloc(size);
}

__device__ void free_wrapper(void* ptr)
{
  if (TheMSys->stats.enabled)
    TheMSys->stats.free++;

  free(ptr);
}

extern "C"
__device__ int init_memsys(void* dummy_return, uint64_t memsys_ptr, bool stats_enabled)
{
  printf("malloc_wrapper address:  %p\n", malloc_wrapper);
  printf("free_wrapper address:  %p\n", free_wrapper);

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
