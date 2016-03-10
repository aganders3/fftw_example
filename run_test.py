import fftw_example
import numpy as np

# import core.math.fft as ft
# A = np.zeros((4,4), dtype=np.complex64)
# A[0,:] = 1
# out_dims = np.array(A.shape, dtype=np.int64)

# ft.fftw(A, out_dims, dim1=0, dim2=1, dir=0)


A = np.zeros((4,4,4), dtype=np.complex64)
A[0,:,:] = 1

fftw_example.guru(A)


