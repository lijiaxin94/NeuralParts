from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

# define an extension that will be cythonized and compiled
ext = Extension(name="mesh_containment_check", 
        sources=["mesh_containment_check.pyx"],
        include_dirs=[numpy.get_include()],
        language='c++')
setup(ext_modules=cythonize(ext))