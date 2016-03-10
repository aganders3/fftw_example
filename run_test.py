# if numpy is imported first, it breaks some of our calls to FFTW
import numpy as np
import fftw_example

A = np.zeros((4,4,4), dtype=np.complex64)
A[0,:,:] = 1

fftw_example.guru(A)

