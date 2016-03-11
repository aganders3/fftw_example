# fftw_example
This repo demonstrates the issues we're having with MKL and FFTW interfering.

Included is a Python script that uses a C-extension to interface with FFTW.
The script will run successfully with `nomkl` version of `numpy`, but fail with the `mkl`-linked version.

Two `conda` environments (`mkl_environment` and `nomkl_environment`) are provided.
Each requires `fftw` - I am using our packaged version in the `gpi` channel on Anaconda.org (https://anaconda.org/GPI/fftw).

You will also need to build the `fftw_example` package itself and install it in each environment:

    python setup.py build install
    
The error is not present when running in the `nomkl` environment:
```
(nomkl)[ash@localhost fftw_example]$ python test_fftw_example.py 
INPUT DATA:
[[[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]]
FFT PLAN:(dft-direct-4-x16 "n1_4")
OUTPUT DATA:
[[[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

 [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

 [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

 [[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]]
```
...but the fftw plan is `NULL` when this is run in the `mkl` environment:
```
(mkl)[ash@localhost fftw_example]$ python test_fftw_example.py 
INPUT DATA:
[[[ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]
  [ 1.+0.j  1.+0.j  1.+0.j  1.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]

 [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
  [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]]
FFTW PLAN IS NULL
This would cause a segfault if we executed the plan.
Instead we will exit gracefully.
```

This also depends on the order in which `numpy` and our C-extension are imported!
If numpy is imported *after* our C-extension, there is no problem.
Unfortunately this workaround is not an option in our project.
