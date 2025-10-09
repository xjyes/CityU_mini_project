import numpy as np
import numpy.linalg as npl

def _chol_solve(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    L = npl.cholesky(A)
    Y = npl.solve(L, B)
    X = npl.solve(L.T, Y)
    return X

def _safe_var(x: np.ndarray) -> float:
    v = np.var(x)
    return float(max(v, 1e-12))

