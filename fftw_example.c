#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include "stdio.h"
#include "fftw3.h"

static PyObject *
fftw_example_guru(PyObject *self, PyObject* args)
{
    PyArrayObject* arr;
    if (!PyArg_ParseTuple(args, "O", &arr))
    {
        return NULL;
    }

    npy_intp* arr_dims = PyArray_DIMS(arr);
    
    int n = arr_dims[0];
    fftwf_complex* arr_ptr = (fftwf_complex*)(PyArray_GETPTR1(arr, 0));
    arr_ptr[0][0] = 1;
    arr_ptr[1][0] = 1;
    arr_ptr[2][0] = 1;
    arr_ptr[3][0] = 1;

    printf("INPUT DATA:\n");
    PyObject_Print((PyObject*)(arr), stdout, Py_PRINT_RAW);
    printf("\n");

    int rank = 1;
    fftwf_iodim guru_dims[rank];
    guru_dims[0].n = n;
    guru_dims[0].is = guru_dims[0].os = n*n;

    int howmany_rank = 2;
    fftwf_iodim howmany_dims[howmany_rank];
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

    if(guru_plan)
    {
        printf("FFT PLAN:");
        fftwf_print_plan(guru_plan);
        printf("\n");
        fftwf_execute(guru_plan);
    }
    else
    {
        printf("FFTW PLAN IS NULL\n");
        printf("This would cause a segfault if we executed the plan.\n");
        printf("Instead we will exit gracefully.\n");
        Py_INCREF(Py_None);
        return Py_None;
    }

    printf("OUTPUT DATA:\n");
    PyObject_Print((PyObject*)(arr), stdout, Py_PRINT_RAW);
    printf("\n");

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

