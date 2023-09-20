from cuda import cudart
from numba import cuda
import numpy as np

ONE_MEGA_BYTE = 1024 * 1024
(ret,) = cudart.cudaDeviceSetLimit(cudart.cudaLimit.cudaLimitMallocHeapSize,
                                   ONE_MEGA_BYTE)
if ret.value != 0:
    raise RuntimeError("Unable to set CUDA malloc heap size")

sizeof_memsys = cuda.declare_device('sizeof_memsys', 'uint64()')
init_memsys = cuda.declare_device('init_memsys', 'void(uint64, bool_)')


@cuda.jit(link=['nrt.ptx'])
def get_msys_size(size_arr):
    size_arr[0] = sizeof_memsys()


size_arr = cuda.device_array(1, np.uint64)

get_msys_size[1, 1](size_arr)
msys_size = size_arr[0]

print(f"Size of NRT_MemSys is {msys_size}")

memsys_holder = cuda.device_array(int(msys_size), np.uint8)
memsys_ptr = memsys_holder.__cuda_array_interface__['data'][0]


def dump_memsys():
    print("Memsys contents:")
    print("".join([f'{v:02x}' for v in memsys_holder[:msys_size]]))


dump_memsys()


@cuda.jit(link=['nrt.ptx'])
def setup_memsys(memsys_ptr):
    init_memsys(memsys_ptr, True)


setup_memsys[1, 1](memsys_ptr)
cuda.synchronize()
dump_memsys()


device_allocate = cuda.declare_device('device_allocate', 'uint64(uint64)')
device_free = cuda.declare_device('device_free', 'void(uint64)')


@cuda.jit(link=['shim.ptx'])
def dynamic_alloc_user():
    ptr = device_allocate(256)
    device_free(ptr)


dynamic_alloc_user.add_global('TheMSys', memsys_ptr)

dynamic_alloc_user[1, 1]()
cuda.synchronize()
dump_memsys()
