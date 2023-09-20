NVCCFLAGS := -gencode arch=compute_75,code=sm_75 -std=c++14 -rdc true -G

all:
	nvcc $(NVCCFLAGS) -ptx nrt.cu
	nvcc $(NVCCFLAGS) -ptx shim.cu
