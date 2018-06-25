""""""

import numpy as np
from numpy.linalg import norm
from numpy.linalg import svd


def rpca(M, mu=None, l=None, tol=1E-7, max_iter=1000):
    """"""
    if not mu:
        mu = norm(M, ord=2, axis=(0,1)) * 1.25

    if not l:
        l = 1 / np.sqrt(np.max(M.shape))

    L = np.zeros(M.shape)
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)

    M_sign = np.sign(M)
    M_sign_norm2 = norm(M_sign, ord=2)
    M_sign_normi = norm(M_sign, ord=np.inf) * (l**-1)

    J = np.max([M_sign_norm2, M_sign_normi])
    Y = M_sign / J

    err = np.inf
    i = 0

    while err > tol and i < max_iter:
        U, S, V = svd(M - S + Y * (mu**-1), full_matrices=False)
        L = np.dot(U, np.dot(np.diag(_shrink(S, mu)), V))
        S = _shrink(M - L + Y * (mu**-1), l * mu)
        Y = Y + mu * (M - L - S)

        err = _calc_error(M, L, S)
        i += 1

    return L, S


def _calc_error(M, L, S):
    """"""
    return norm(M - L - S, ord='fro') / norm(M, ord='fro')


def _shrink(M, t):
    """"""
    return np.sign(M) * np.maximum((np.abs(M) - t), np.zeros(M.shape))
