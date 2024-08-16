import subprocess
from setuptools import Extension, setup
from Cython.Build import cythonize

import numpy

subprocess.run("make")

setup(
    name='tiny-torch',
    ext_modules=cythonize([
    Extension("tinytorch", ["engine.pyx"],
              include_dirs=[numpy.get_include()],
              libraries=["bin/cpu_backend.so"])
    ], annotate=True),
)