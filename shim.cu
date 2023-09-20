#include "nrt.cuh"
#include <stdio.h>

extern "C" __device__ int
device_allocate(intptr_t* return_value, size_t size)
{
  printf("memsys from shim: 0x%p\n", TheMSys);
  //printf("allocator from shim: 0x%p\n", TheMSys->allocator.malloc);
  void* ptr = TheMSys->allocator.malloc(size);
  *return_value = reinterpret_cast<intptr_t>(ptr);
  return 0;
}

extern "C" __device__ int
device_free(uint64_t* dummy_return, intptr_t ptr)
{
  TheMSys->allocator.free(reinterpret_cast<void*>(ptr));
  return 0;
}
