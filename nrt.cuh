#pragma once

#include <cuda/atomic>

typedef void* (*NRT_malloc_func)(size_t size);
typedef void* (*NRT_realloc_func)(void* ptr, size_t new_size);
typedef void (*NRT_free_func)(void* ptr);
  
// Globally needed variables
struct NRT_MemSys {
  /* System allocation functions */
  struct {
    NRT_malloc_func malloc;
    NRT_realloc_func realloc;
    NRT_free_func free;
  } allocator;
  struct {
    bool enabled;
    cuda::atomic<size_t, cuda::thread_scope_device> alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> free;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_alloc;
    cuda::atomic<size_t, cuda::thread_scope_device> mi_free;
  } stats;
};

extern __device__ NRT_MemSys *TheMSys;


