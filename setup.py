import subprocess
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

subprocess.run("make clean", shell=True)
subprocess.run("make all", shell=True)

# Main engine extension
engine = Extension(
    name="tinytorch",
    sources=["src/tinytorch.pyx"],
    include_dirs=[np.get_include(), '.'],
    library_dirs=["."],
    libraries=['cudart'],
    extra_objects=["build/cuda_backend.so", "build/memutils.so", "build/cpu_backend.so"],
    language="c++",
    extra_compile_args={},
    runtime_library_dirs=["."]
)

setup(
    name='tinytorch',
    ext_modules=cythonize([engine], annotate=True),
    include_dirs=[np.get_include(), ".", "src"]
)
