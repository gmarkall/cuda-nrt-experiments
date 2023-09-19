NVCCFLAGS := -gencode arch=compute_75,code=sm_75 -std=c++14 -rdc true

all:
	nvcc $(NVCCFLAGS) -ptx nrt.cu
