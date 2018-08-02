"""
Robust Principal Component Analysis
"""

import numpy as np

from numpy.linalg import norm
from numpy.linalg import svd


def rpca_alm(M, mu=None, l=None, tol=1E-7, max_iter=1000):
    """Matrix recovery/decomposition using Robust Principal Component Analysis
    with Augmented Lagrangian Method (ALM)

    Decompose a rectengular matrix M into a low-rank component, and a sparse
    component, by solving a convex program called Principal Component Pursuit.

    minimize        ||A||_* + λ ||E||_1
    subject to      A + E = M

    where           ||A||_* is the nuclear norm of A (sum of singular values)
                    ||E||_1 is the l1 norm of E (absolute values of elements)

    Parameters
    ----------

    M : array-like, shape (n_samples, n_features)
        Matrix to decompose, where n_samples in the number of samples and
        n_features is the number of features.

    mu : float (default 1.25 * ||M||_2)
        Parameter from the Augmented Lagrange Multiplier form of Principal
        Component Pursuit (PCP). [2]_

    l : float (default 1/sqrt(max(m,n)), for m x n of M)
        Parameter of the convex problem ||A||_* + l ||E||_1. [2]_

    tol : float >= 0 (default 1E-7)
        Tolerance for accuracy of matrix reconstruction of low rank and sparse
        components.

    max_iter : int >= 0 (default 1000)
        Maximum number of iterations to perform.


    Returns
    -------

    A : array, shape (n_samples, n_features)
        Low-rank component of the matrix decomposition.

    E : array, shape (n_samples, n_features)
        Sparse component of the matrix decomposition.

    err : float
        Error of matrix reconstruction

    References
    ----------

    .. [1] Z. Lin, M. Chen, Y. Ma. The Augmented Lagrange Multiplier Method for
           Exact Recovery of Corrupted Low-Rank Matrices, arXiv:1009.5055

    .. [2] E. J. Candés, X. Li, Y. Ma, J. Wright. Robust principal
           component analysis? Journal of the ACM v.58 n.11 May 2011

    """

    rho = 1.5

    if not mu:
        mu = 1.25 * norm(M, ord=2)

    mutol = 1E7

    if not l:
        l = np.max(M.shape)**-.5

    M_sign = np.sign(M)
    norm_spectral = norm(M_sign, ord=2)
    norm_inf = norm(M_sign, ord=np.inf)
    norm_dual = np.max([norm_spectral, norm_inf * l**-1])

    Y = M_sign * norm_dual**-1
    A = np.zeros(M.shape)
    E = np.zeros(M.shape)

    err = np.inf
    i = 0

    while err > tol and i < max_iter:
        U, S, V = svd(M - E + Y * mu**-1, full_matrices=False)

        A = np.dot(U, np.dot(np.diag(_shrink(S, mu**-1)), V))
        E = _shrink(M - A + Y * mu**-1, l * mu**-1)
        Y = Y + mu * (M - A - E)

        err = _fro_error(M, A, E)
        mu *= rho
        mu = np.min([mu, mutol])
        i += 1

    return A, E, err


def _fro_error(M, A, E):
    """Error of matrix reconstruction"""
    return norm(M - A - E, ord='fro') / norm(M, ord='fro')


def _shrink(M, t):
    """Shrinkage operator"""
    return np.sign(M) * np.maximum((np.abs(M) - t), np.zeros(M.shape))
