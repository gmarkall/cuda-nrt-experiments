#include "nrt.cuh"

extern "C" __device__ int
device_allocate(intptr_t* return_value, size_t size)
{
  *return_value = reinterpret_cast<intptr_t>(TheMSys->allocator.malloc(size));
  return 0;
}

extern "C" __device__ int
device_free(uint64_t* dummy_return, intptr_t ptr)
{
  TheMSys->allocator.free(reinterpret_cast<void*>(ptr));
  return 0;
}
