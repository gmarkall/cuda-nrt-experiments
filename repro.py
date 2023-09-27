from cuda import cudart
from numba import cuda
import numpy as np
import os
import re

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


print(f"0x{memsys_ptr:x}")
memsys_ptr_byte_strings = re.findall('..', f"{memsys_ptr:016x}")
prefixed_reversed = [str(int(x, 16)) for x in reversed(memsys_ptr_byte_strings)]
ptx_memsys_array_literal = "{" + ", ".join(prefixed_reversed) + "}"

memsys_ptx = f"""\
.version 8.2
.target sm_75
.address_size 64

.visible .global .align 8 .b8 TheMSys[8] = {ptx_memsys_array_literal};
"""

print(memsys_ptx)

memsys_ptx_file = "memsys.ptx"

try:
    os.remove(memsys_ptx_file)
except FileNotFoundError:
    pass

with open(memsys_ptx_file, 'w') as f:
    f.write(memsys_ptx)


@cuda.jit(link=['shim.ptx', 'memsys.ptx'], debug=True)
def dynamic_alloc_user():
    ptr = device_allocate(256)
    print(ptr)
    device_free(ptr)


dynamic_alloc_user[1, 1]()
cuda.synchronize()
dump_memsys()
