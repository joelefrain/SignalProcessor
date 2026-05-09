# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport fabs


def trapezoid_integrate(cnp.ndarray[cnp.double_t, ndim=1] values, double dt, double init=0.0):
    cdef Py_ssize_t n = values.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    cdef double half_dt = 0.5 * dt
    out[0] = init
    for i in range(1, n):
        out[i] = out[i - 1] + (values[i - 1] + values[i]) * half_dt
    return out


def central_difference(cnp.ndarray[cnp.double_t, ndim=1] values, double dt):
    cdef Py_ssize_t n = values.shape[0]
    cdef cnp.ndarray[cnp.double_t, ndim=1] out = np.empty(n, dtype=np.float64)
    cdef Py_ssize_t i
    if n == 1:
        out[0] = 0.0
        return out
    if n < 5:
        out[0] = (values[1] - values[0]) / dt
        for i in range(1, n - 1):
            out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt)
        out[n - 1] = (values[n - 1] - values[n - 2]) / dt
        return out

    out[0] = (values[1] - values[0]) / dt
    out[1] = (values[2] - values[0]) / (2.0 * dt)
    for i in range(2, n - 2):
        out[i] = (values[i - 2] - 8.0 * values[i - 1] + 8.0 * values[i + 1] - values[i + 2]) / (12.0 * dt)
    out[n - 2] = (values[n - 1] - values[n - 3]) / (2.0 * dt)
    out[n - 1] = (values[n - 1] - values[n - 2]) / dt
    return out
