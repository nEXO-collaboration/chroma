import numpy as np
import pytools
import pycuda.tools
from pycuda import characterize
import pycuda.driver as cuda
import pycuda.compiler
from pycuda import gpuarray as ga

from chroma.cuda import srcdir

# standard nvcc options
cuda_options = ('--use_fast_math',)#, '--ptxas-options=-v']

@pycuda.tools.context_dependent_memoize
def get_cu_module(name, options=None, include_source_directory=True):
    """Returns a pycuda.compiler.SourceModule object from a CUDA source file
    located in the chroma cuda directory at cuda/[name]."""
    if options is None:
        options = []
    elif isinstance(options, tuple):
        options = list(options)
    else:
        raise TypeError('`options` must be a tuple.')

    if include_source_directory:
        options += ['-I' + srcdir]

    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()

    return pycuda.compiler.SourceModule(source, options=options,
                                        no_extern_c=True)

@pytools.memoize
def get_cu_source(name):
    """Get the source code for a CUDA source file located in the chroma cuda
    directory at src/[name]."""
    with open('%s/%s' % (srcdir, name)) as f:
        source = f.read()
    return source

class GPUFuncs(object):
    """Simple container class for GPU functions as attributes."""
    def __init__(self, module):
        self.module = module
        self.funcs = {}

    def __getattr__(self, name):
        try:
            return self.funcs[name]
        except KeyError:
            f = self.module.get_function(name)
            self.funcs[name] = f
            return f

init_rng_src = """
#include <curand_kernel.h>

extern "C"
{

__global__ void init_rng(int nthreads, curandState *s, unsigned long long seed, unsigned long long offset)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if (id >= nthreads)
		return;

	curand_init(seed, id, offset, &s[id]);
}

} // extern "C"
"""

def get_rng_states(size, seed=1):
    "Return `size` number of CUDA random number generator states."
    rng_states = cuda.mem_alloc(size*characterize.sizeof('curandStateXORWOW', '#include <curand_kernel.h>'))

    module = pycuda.compiler.SourceModule(init_rng_src, no_extern_c=True)
    init_rng = module.get_function('init_rng')

    init_rng(np.int32(size), rng_states, np.uint64(seed), np.uint64(0), block=(64,1,1), grid=(size//64+1,1))

    return rng_states

def to_float3(arr):
    "Returns an pycuda.gpuarray.vec.float3 array from an (N,3) array."
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError('to_float3 expects array size divisible by 3.')
        arr = arr.reshape((-1, 3))
    elif arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError('to_float3 expects shape (N,3) for array inputs.')
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.asarray(arr, order='c')
    return arr.astype(np.float32, copy=False).view(ga.vec.float3).reshape(-1)

def to_uint3(arr):
    "Returns a pycuda.gpuarray.vec.uint3 array from an (N,3) array."
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.size % 3 != 0:
            raise ValueError('to_uint3 expects array size divisible by 3.')
        arr = arr.reshape((-1, 3))
    elif arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError('to_uint3 expects shape (N,3) for array inputs.')
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.asarray(arr, order='c')
    return arr.astype(np.uint32, copy=False).view(ga.vec.uint3).reshape(-1)

def chunk_iterator(nelements, nthreads_per_block=64, max_blocks=1024):
    """Iterator that yields tuples with the values requried to process
    a long array in multiple kernel passes on the GPU.

    Each yielded value is of the form,
        (first_index, elements_this_iteration, nblocks_this_iteration)

    Example:
        >>> list(chunk_iterator(300, 32, 2))
        [(0, 64, 2), (64, 64, 2), (128, 64, 2), (192, 64, 2), (256, 9, 1)]
    """
    first = 0
    while first < nelements:
        elements_left = nelements - first
        blocks = int(elements_left // nthreads_per_block)
        if elements_left % nthreads_per_block != 0:
            blocks += 1 # Round up only if needed
        blocks = min(max_blocks, blocks)
        elements_this_round = min(elements_left, blocks * nthreads_per_block)

        yield (first, elements_this_round, blocks)
        first += elements_this_round

def create_cuda_context(device_id=None):
    """Initialize and return a CUDA context on the specified device.
    If device_id is None, the default device is used."""
    if device_id is None:
        try:
            context = pycuda.tools.make_default_context()
        except cuda.LogicError:
            # initialize cuda
            cuda.init()
            context = pycuda.tools.make_default_context()
    else:
        try:
            device = cuda.Device(device_id)
        except cuda.LogicError:
            # initialize cuda
            cuda.init()
            device = cuda.Device(device_id)
        context = device.make_context()

    context.set_cache_config(cuda.func_cache.PREFER_L1)

    return context

vec_dtypes = set([ x for x in list(ga.vec.__dict__.values()) if isinstance(x,np.dtype) ])

def make_gpu_struct(size, members):
    struct = cuda.mem_alloc(size)

    i = 0
    for member in members:
        if isinstance(member, ga.GPUArray):
            member = member.gpudata

        if isinstance(member, cuda.DeviceAllocation):
            if i % 8:
                raise Exception('cannot align 64-bit pointer. '
                                'arrange struct member variables in order of '
                                'decreasing size.')

            cuda.memcpy_htod(int(struct)+i, np.intp(int(member)).data)
            i += 8
        elif np.isscalar(member) or (hasattr(member, 'dtype') and member.dtype in vec_dtypes and member.shape == ()):
            cuda.memcpy_htod(int(struct)+i, member.data)
            i += member.nbytes
        else:
            raise TypeError('expected a GPU device pointer or scalar type.')

    return struct

def format_size(size):
    if size < 1e3:
        return '%.1f%s' % (size, ' ')
    elif size < 1e6:
        return '%.1f%s' % (size/1e3, 'K')
    elif size < 1e9:
        return '%.1f%s' % (size/1e6, 'M')
    else:
        return '%.1f%s' % (size/1e9, 'G')

def format_array(name, array):
    return '%-15s %6s %6s' % \
        (name, format_size(len(array)), format_size(array.nbytes))

def Mapped(array):
    '''Analog to pycuda.driver.InOut(), but indicates this array
    is memory mapped to the device space and should not be copied.

    To simplify coding, Mapped() will pass anything with a gpudata
    member, like a gpuarray, through unchanged.
    '''
    if hasattr(array, 'gpudata'):
        return array
    else:
        return np.intp(array.base.get_device_pointer())

def mapped_alloc(pagelocked_alloc_func, shape, dtype, write_combined):
    '''Returns a pagelocked host array mapped into the CUDA device
    address space, with a gpudata field set so it just works with CUDA 
    functions.'''
    flags = cuda.host_alloc_flags.DEVICEMAP
    if write_combined:
        flags |= cuda.host_alloc_flags.WRITECOMBINED
    array = pagelocked_alloc_func(shape=shape, dtype=dtype, mem_flags=flags)
    return array

def mapped_empty(shape, dtype, write_combined=False):
    '''See mapped_alloc()'''
    return mapped_alloc(cuda.pagelocked_empty, shape, dtype, write_combined)

def mapped_empty_like(other, write_combined=False):
    '''See mapped_alloc()'''
    return mapped_alloc(cuda.pagelocked_empty, other.shape, other.dtype,
                        write_combined)

def mapped_zeros(shape, dtype, write_combined=False):
    '''See mapped_alloc()'''
    return mapped_alloc(cuda.pagelocked_zeros, shape, dtype, write_combined)

def mapped_zeros_like(other, write_combined=False):
    '''See mapped_alloc()'''
    return mapped_alloc(cuda.pagelocked_zeros, other.shape, other.dtype,
                        write_combined)
