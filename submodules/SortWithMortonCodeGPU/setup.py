from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

os.path.dirname(os.path.abspath(__file__))

setup(
    name="cuda_sort_lib",
    ext_modules=[
        CUDAExtension(
            name='cuda_lib',
            sources=[
                './src/ComputeMortonCode.cu',
                './src/SortByMortonCode.cu',
                './ext.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)
