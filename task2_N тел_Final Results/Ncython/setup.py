from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
# ext=Extension(name='fact', sources=['test2.pyx'])
# setup(ext_modules=cythonize(ext))

# setup(
#     ext_modules=cythonize("test2.pyx"),
#     zip_safe=False,)
setup(ext_modules = cythonize('cysolar.pyx'), include_dirs=[numpy.get_include()])