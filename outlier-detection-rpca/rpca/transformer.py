""""""

from sklearn.base import TransformerMixin

from algorithm import rpca


class RobustPCA(TransformerMixin):
    def __init__(self, method='sparse',
                 mu=None,
                 l=None,
                 max_iter=1000,
                 tol=1E-7):

        if not (method == 'sparse' or method == 'lowrank'):
            raise ValueError('method must be either `sparse` or `lowrank`')

        self._method = method
        self._mu = mu
        self._l = l
        self._tol = tol

    def transform(self, X):
        self.L_, self.S_ = rpca(X, mu=self._mu, l=self._l, tol=self._tol)
        if self._method == 'sparse':
            return self.S_
        elif self._method == 'lowrank':
            return self.L_
        else:
            raise ValueError('unknown method')
