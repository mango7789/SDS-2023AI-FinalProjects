from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "cy_graph3",
        ["cy_graph3.py"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),     Extension(
        "labeler",
        ["labeler.py"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(ext_modules=cythonize(ext_modules),
      include_dirs=[numpy.get_include()])
