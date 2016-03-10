from distutils.core import setup, Extension
import numpy

module1 = Extension('fftw_example',
                    include_dirs = [numpy.get_include(), '/home/ash/gpi_stack/include'],
                    libraries = ['fftw3', 'fftw3f'],
                    sources = ['fftw_example.c'])

setup (name = 'FFTW_Example',
       version = '0.0',
       description = 'This package demonstrates an issue with FFTW and MKL',
       ext_modules = [module1])
