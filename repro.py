from numba import cuda
import numpy as np

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


@cuda.jit(link=['nrt.ptx'])
def setup_memsys(memsys_ptr):
    init_memsys(memsys_ptr, True)


setup_memsys[1, 1](memsys_ptr)

cuda.synchronize()

print("Memsys contents:")
print("".join([f'{v:02x}' for v in memsys_holder[:msys_size]]))
