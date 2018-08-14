"""
Robust Principal Component Analysis Scikit-Learn transformer
"""

from sklearn.base import TransformerMixin

from algorithm import rpca_alm


class RobustPCA(TransformerMixin):
    """Matrix recovery/decomposition using Robust Principal Component Analysis

    Decompose a rectengular matrix M into a low-rank component, and a sparse
    component, by solving a convex minimization problem via Augmented Lagrangian
    Method.

    minimize        ||A||_* + λ ||E||_1
    subject to      A + E = M

    where           ||A||_* is the nuclear norm of A (sum of singular values)
                        - surrogate of matrix rank
                    ||E||_1 is the l1 norm of E (absolute values of elements)
                        - surrogate of matrix sparseness

    Relaxed to

        L(A,E,Y,λ) .= ||A||_* + λ||E||_1 + <Y, M-A-E> + µ/2 ||M-A-E||_F^2

    Parameters
    ----------

    M : array-like, shape (n_samples, n_features)
        Matrix to decompose, where n_samples in the number of samples and
        n_features is the number of features.

    l : float (default 1/sqrt(max(m,n)), for m x n of M)
        Parameter (lambda) of the convex problem ||A||_* + λ ||E||_1. [2]_

    mu : float (default 1.25 * ||M||_2)
        Parameter (mu) from the Augmented Lagrange Multiplier form of Principal
        Component Pursuit (PCP). [2]_

    mu_tol : float >= 0 (default 1E-7)
        Weight parameter.

    tol : float >= 0 (default 1E-7)
        Tolerance for accuracy of matrix reconstruction of low rank and sparse
        components.

    max_iter : int >= 0 (default 1000)
        Maximum number of iterations to perform.

    Attributes
    ----------

    low_rank_ : array, (n_samples, n_features)
        Low-rank component of the matrix decomposition.

    sparse_ : array, (n_samples, n_features)
        Sparse component of the matrix decomposition.

    error_ : float
        Error of matrix reconstruction
        ||M-A-E||_F / ||M||_F

    References
    ----------

    .. [1] Z. Lin, M. Chen, Y. Ma. The Augmented Lagrange Multiplier Method for
           Exact Recovery of Corrupted Low-Rank Matrices, arXiv:1009.5055

    .. [2] E. J. Candés, X. Li, Y. Ma, J. Wright. Robust principal
           component analysis? Journal of the ACM v.58 n.11 May 2011

    """

    def __init__(self, method='sparse',
                 l=None,
                 mu=None,
                 mu_tol=1E7,
                 tol=1E-7,
                 max_iter=1000):

        options = ['sparse', 'low_rank']

        if method not in options:
            raise ValueError(f'method must be one of {options}')

        self._method = method
        self._l = l
        self._mu = mu
        self._mu_tol = mu_tol
        self._tol = tol

        self.low_rank_ = None
        self.sparse_ = None
        self.error_ = None

    def transform(self, X):
        """Matrix decomposition/recovery with RPCA and ALM

        Parameters
        ----------

        X : array-like, shape (n_samples, n_features)
            Matrix to decompose, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------

        X_new : array-like, shape (n_samples, n_features)
            Decomposed matrix, either sparse component (method `sparse`) or
            low-rank component (method `low_rank`)
        """
        self.low_rank_, self.sparse_, self.error_ = rpca_alm(X,
                                                             mu=self._mu,
                                                             l=self._l,
                                                             mu_tol=self._mu_tol,
                                                             tol=self._tol)
        if self._method == 'sparse':
            return self.sparse_
        elif self._method == 'low_rank':
            return self.low_rank_
        else:
            raise ValueError('unknown method')
