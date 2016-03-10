#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "stdio.h"
#include "fftw3.h"

static PyObject *
fftw_example_guru(PyObject *self, PyObject* args)
{
    int n = 2;
    npy_intp dims[3] = {n, n, n};
    PyArrayObject* arr = (PyArrayObject*)PyArray_ZEROS(3, dims, NPY_COMPLEX64, 0);
    PyObject_Print(PyArray_BASE(arr), stdout, Py_PRINT_RAW);
    printf("\n");

    fftwf_complex* arr_ptr = (fftwf_complex*)(PyArray_GetPtr(arr, 0));

    int rank = 1;
    fftw_iodim guru_dims[rank];
    guru_dims[0].n = n;
    guru_dims[0].is = guru_dims[0].os = n*n;

    int howmany_rank = 2;
    fftw_iodim howmany_dims[howmany_rank];
    howmany_dims[0].n = n;
    howmany_dims[0].is = howmany_dims[0].os = 1;
    howmany_dims[1].n = n;
    howmany_dims[1].is = howmany_dims[1].os = n;
    howmany_dims[2].n = n;
    howmany_dims[2].is = howmany_dims[2].os = n*n;

    // generate plan according to fftw guru interface
    // transform each col of a 3d array
    fftwf_plan guru_plan = fftwf_plan_guru_dft(rank, guru_dims,
                                              howmany_rank, howmany_dims,
                                              arr_ptr, arr_ptr,
                                              FFTW_FORWARD, FFTW_ESTIMATE);

    fftwf_print_plan(guru_plan);

    Py_INCREF(Py_None);
    return Py_None;
}

static PyMethodDef fftw_exampleMethods[] = {
    {"guru", fftw_example_guru, METH_VARARGS, "Test the FFTW guru interface."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fftw_examplemodule= {
   PyModuleDef_HEAD_INIT,
   "fftw_example",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    fftw_exampleMethods
};

PyMODINIT_FUNC
PyInit_fftw_example(void)
{
    PyObject* m = PyModule_Create(&fftw_examplemodule);
    import_array();
    return m;
}

