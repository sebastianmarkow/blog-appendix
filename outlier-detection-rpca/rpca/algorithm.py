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

    Y = np.zeros(M.shape)
    A = np.zeros(M.shape)
    E = np.zeros(M.shape)

    M_sign = np.sign(M)
    M_sign_norm2 = norm(M_sign, ord=2)
    M_sign_normi = norm(M_sign, ord=np.inf) * (l**-1)

    J = np.max([M_sign_norm2, M_sign_normi])
    Y = M_sign / J

    err = np.inf
    i = 0

    while err > tol and i < max_iter:
        U, S, V = svd(M - E + Y * (mu**-1), full_matrices=False)
        A = np.dot(U, np.dot(np.diag(_shrink(S, mu)), V))
        E = _shrink(M - A + Y * (mu**-1), l * mu)
        Y = Y + mu * (M - A - E)

        err = _calc_error(M, A, E)
        i += 1

    return A, E


def _calc_error(M, A, E):
    """"""
    return norm(M - A - E, ord='fro') / norm(M, ord='fro')


def _shrink(M, t):
    """"""
    return np.sign(M) * np.maximum((np.abs(M) - t), np.zeros(M.shape))
