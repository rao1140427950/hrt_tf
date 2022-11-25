import numpy as np
from _lasso import lasso_solve

ENABLE_WARNING = False

def solve(A: np.ndarray, b: np.ndarray, lam: float):
    """
    Solve a LASSO problem: minimize ||Ax - b||2 + lambda * ||x||1
    :param A: ndarray of shape (m, n) and type float64
    :param b: ndarray of shape (m, 1) or (m,) and type float64
    :param lam: float.
    :return: x: ndarray of shape (n, 1) and type float64
    """
    m, n = np.shape(A)
    if b.shape != (m, 1) and b.shape != (m,):
        raise ValueError("The shape of Matrix `A` and Vector 'b' are incompatible.")
    if A.dtype != np.float64 and ENABLE_WARNING:
        print("Warining: convert `A` to float64.")
        A = A.astype(np.float64)
    if b.dtype != np.float64 and ENABLE_WARNING:
        print("Warining: convert `b` to float64.")
        b = b.astype(np.float64)
    rdict = lasso_solve(A.reshape((m * n,)), m, n, b.reshape(m,), lam)
    x = rdict['x']
    status = rdict['status']
    if status != 0  and ENABLE_WARNING:
        print("Warning: solved with status {}.".format(status))
    return x

def norm_sp(x):
    s = np.sum(x)
    x /= s
    return x

def resp_lasso(t_mat, mea_mat, df, lam=100.):
    D_mul = np.matmul(t_mat, df)
    I_out = solve(D_mul, mea_mat, lam)
    resp = np.matmul(df, I_out)

    return resp
