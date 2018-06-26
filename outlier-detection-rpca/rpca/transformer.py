""""""

from sklearn.base import TransformerMixin

from algorithm import rpca


class RobustPCA(TransformerMixin):
    def __init__(self, method='sparse',
                 mu=None,
                 l=None,
                 max_iter=1000,
                 tol=1E-7):

        if not (method == 'sparse' or method == 'recovery'):
            raise ValueError('method must be either `sparse` or `recovery`')

        self._method = method
        self._mu = mu
        self._l = l
        self._tol = tol

    def transform(self, X):
        self.A_, self.E_ = rpca(X, mu=self._mu, l=self._l, tol=self._tol)
        if self._method == 'sparse':
            return self.E_
        elif self._method == 'recovery':
            return self.A_
        else:
            raise ValueError('unknown method')
