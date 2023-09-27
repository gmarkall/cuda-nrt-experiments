#include <iostream>
#include <cuda.h>

__global__ void allocate(int **p)
{
  *p = static_cast<int*>(malloc(sizeof(int)));
}

__global__ void deallocate(int *p)
{
  free(p);
}

__managed__ int *p;

int main(int argc, char **argv)
{
  cudaError_t err;
  cudaSetDevice(0);
  constexpr size_t ONE_MEGABYTE = 1024 * 1024;
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, ONE_MEGABYTE);
  std::cout << "Allocating... ";
  allocate<<<1, 1>>>(&p);
  err = cudaDeviceSynchronize();
  std::cout << "result: '" << cudaGetErrorString(err) << "'" << std::endl;
  std::cout << "Deallocating... ";
  deallocate<<<1, 1>>>(p);
  err = cudaDeviceSynchronize();
  std::cout << "result: '" << cudaGetErrorString(err) << "'" << std::endl;
  return 0;
}
