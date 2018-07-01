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

    if not mu:
        mu = 1.25 * norm(M, ord=2, axis=(0,1))

    if not l:
        l = 1 / np.sqrt(np.max(M.shape))

    Y = np.zeros(M.shape)
    A = np.zeros(M.shape)
    E = np.zeros(M.shape)

    M_sgn = np.sign(M)
    M_spectral = norm(M_sgn, ord=2)
    M_absolute = norm(M_sgn, ord=np.inf) * (l**-1)

    Y = M_sgn / np.max([M_spectral, M_absolute])

    err = np.inf
    i = 0

    while err > tol and i < max_iter:
        U, S, V = svd(M - E + Y * (mu**-1), full_matrices=False)

        A = np.dot(U, np.dot(np.diag(_shrink(S, mu)), V))
        E = _shrink(M - A + Y * (mu**-1), l * mu)
        Y = Y + mu * (M - A - E)

        err = _fro_error(M, A, E)
        i += 1

    return A, E, err


def _fro_error(M, A, E):
    """Error of matrix reconstruction"""
    return norm(M - A - E, ord='fro') / norm(M, ord='fro')


def _shrink(M, t):
    """Shrinkage operator"""
    return np.sign(M) * np.maximum((np.abs(M) - t), np.zeros(M.shape))
