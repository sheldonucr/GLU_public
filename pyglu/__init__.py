"""
pyglu — Python bindings for the GLU GPU-accelerated sparse LU solver.

API mirrors scipy.sparse.linalg.splu / spsolve:

    lu = pyglu.splu(A)          # factorize (returns GLUFactorization)
    x  = lu.solve(b)            # solve Ax = b

    x  = pyglu.spsolve(A, b)    # factorize and solve in one call

A can be any scipy sparse matrix (any format) or a
(data, indices, indptr, shape) tuple in CSC format.

Note: GLU uses single-precision (float32) arithmetic internally on the GPU.
The Python interface accepts and returns float64 arrays, converting at the
boundary. Residuals will reflect single-precision accuracy (~1e-5 relative).
"""

import numpy as np

try:
    import scipy.sparse as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from ._pyglu import GLUFactorization


def _to_csc_arrays(A):
    """Return (data_f64, indices_i32, indptr_i32, n) from a sparse matrix."""
    if _HAS_SCIPY and _sp.issparse(A):
        A = A.tocsc()
        A.sum_duplicates()
        A.sort_indices()
        n = A.shape[0]
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"Matrix must be square, got shape {A.shape}")
        data    = np.ascontiguousarray(A.data,    dtype=np.float64)
        indices = np.ascontiguousarray(A.indices, dtype=np.int32)
        indptr  = np.ascontiguousarray(A.indptr,  dtype=np.int32)
    elif isinstance(A, tuple) and len(A) == 4:
        data, indices, indptr, shape = A
        n = shape[0]
        if shape[0] != shape[1]:
            raise ValueError(f"Matrix must be square, got shape {shape}")
        data    = np.ascontiguousarray(data,    dtype=np.float64)
        indices = np.ascontiguousarray(indices, dtype=np.int32)
        indptr  = np.ascontiguousarray(indptr,  dtype=np.int32)
    else:
        raise TypeError(
            "A must be a scipy sparse matrix or a "
            "(data, indices, indptr, shape) tuple in CSC format"
        )

    if len(indptr) != n + 1:
        raise ValueError(
            f"indptr length {len(indptr)} inconsistent with n={n}"
        )
    if n == 0:
        raise ValueError("Matrix dimension must be > 0")

    return data, indices, indptr, n


def splu(A, perturb=False):
    """Compute the sparse LU factorization of A on the GPU.

    Parameters
    ----------
    A : scipy.sparse matrix (any format) or (data, indices, indptr, shape) tuple
        Square n-by-n sparse matrix.
    perturb : bool, optional
        Enable diagonal perturbation (GESP) to handle near-singular pivots.

    Returns
    -------
    factorization : GLUFactorization
        Object with a ``.solve(b)`` method and a ``.n`` attribute.

    Raises
    ------
    ValueError
        If the matrix is not square or arrays are inconsistent.
    RuntimeError
        If preprocessing or GPU factorization fails.

    Examples
    --------
    >>> import scipy.sparse as sp, numpy as np, pyglu
    >>> A = sp.eye(4, format='csc') * 2.0
    >>> lu = pyglu.splu(A)
    >>> x = lu.solve(np.ones(4))
    """
    data, indices, indptr, n = _to_csc_arrays(A)
    return GLUFactorization(data, indices, indptr, n, perturb)


def spsolve(A, b, perturb=False):
    """Solve the sparse linear system Ax = b on the GPU.

    Parameters
    ----------
    A : scipy.sparse matrix or (data, indices, indptr, shape) tuple
        Square n-by-n coefficient matrix.
    b : array_like, shape (n,)
        Right-hand side vector.
    perturb : bool, optional
        Enable diagonal perturbation. Default False.

    Returns
    -------
    x : numpy.ndarray, float64, shape (n,)

    Examples
    --------
    >>> import scipy.sparse as sp, numpy as np, pyglu
    >>> A = sp.eye(4, format='csc') * 2.0
    >>> x = pyglu.spsolve(A, np.ones(4))
    """
    b = np.asarray(b, dtype=np.float64)
    return splu(A, perturb=perturb).solve(b)


__all__ = ["splu", "spsolve", "GLUFactorization"]
