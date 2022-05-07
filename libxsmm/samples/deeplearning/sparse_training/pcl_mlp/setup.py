import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# Surface
#LIBXSMM_ROOT_PATH="/home/kaotixs/libxsmm_project/libxsmm/"

# Azure
LIBXSMM_ROOT_PATH="/home/kshvedov/libxsmm_project/libxsmm/"

# PC
#LIBXSMM_ROOT_PATH="/root/libxsmm_project/libxsmm/"

setup(name='pcl_mlp',
      py_modules = ['pcl_mlp'],
      ext_modules=[CppExtension('pcl_mlp_ext', ['pcl_mlp_ext.cpp'], extra_compile_args=['-fopenmp', '-g', '-march=native'],
        include_dirs=['{}/include/'.format(LIBXSMM_ROOT_PATH)],
        library_dirs=['{}/lib/'.format(LIBXSMM_ROOT_PATH)],
        libraries=['xsmm'])],
      cmdclass={'build_ext': BuildExtension})

