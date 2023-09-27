STANDARDFLAGS := -gencode arch=compute_75,code=sm_75 -std=c++14
NVCCFLAGS := $(STANDARDFLAGS) -rdc true -G


all:
	nvcc $(NVCCFLAGS) -ptx nrt.cu
	nvcc $(NVCCFLAGS) -ptx shim.cu
	nvcc $(NVCCFLAGS) -ptx malloc_use.cu
	nvcc $(NVCCFLAGS) -dc malloc_use.cu
	nvcc $(NVCCFLAGS) -dlink malloc_use.cu -o malloc_use.bin
	nvcc $(STANDARDFLAGS) -g -G -o malloc_test_harness malloc_test_harness.cu
