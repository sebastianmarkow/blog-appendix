
import numpy as np

from pyspark import keyword_only
from pyspark.ml import Estimator
from pyspark.ml.param import Params
from pyspark.ml.param import TypeConverters
from pyspark.ml.param.shared import HasInputCol
from pyspark.ml.param.shared import HasOutputCol
from pyspark.ml.param.shared import Param
from pyspark.mllib.linalg.distributed import IndexedRowMatrix
from pyspark.mllib.linalg.distributed import IndexedRow

class RPCA(Estimator, HasInputCol, HasOutputCol):
    
    method = Param(Params._dummy(), 'method', 'Output of rpca', typeConverter=TypeConverters.toString)
    mu = Param(Params._dummy(), 'mu', 'Parameter from the Augmented Lagrange Multiplier form', typeConverter=TypeConverters.toFloat)
    l = Param(Params._dummy(), 'l', 'Parameter mix', typeConverter=TypeConverters.toFloat)
    tol = Param(Params._dummy(), 'tol', 'Tolerance accuracy of matrix reconstruction', typeConverter=TypeConverters.toFloat)
    max_iter = Param(Params._dummy(), 'max_iter', 'Maximum number of iterations', typeConverter=TypeConverters.toInt)
    indexCol = Param(Params._dummy(), 'indexCol', 'Column used to index', typeConverter=TypeConverters.toString)
    
    @keyword_only
    def __init__(self, method='sparse', mu=None, l=None, tol=1E-7, max_iter=1000, inputCol=None, outputCol=None):
        super(RPCA, self).__init__()
        self._setDefault(method='sparse', mu=None, l=None, tol=1E-7, max_iter=1000, inputCol=None, outputCol=None)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
    
    @keyword_only
    def setParams(self, method='sparse', mu=None, l=None, tol=1E-7, max_iter=1000, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
    
    def setMethod(self, value):
        return self._set(method=value)
    
    def getMethod(self):
        return self.getOrDefault(self.method)
      
    def setMu(self, value):
        return self._set(mu=value)
    
    def getMu(self):
        return self.getOrDefault(self.mu)
        
    def setL(self, value):
        return self._set(l=value)
    
    def getL(self):
        return self.getOrDefault(self.l)
        
    def setTol(self, value):
        return self._set(tol=value)
    
    def getTol(self):
        return self.getOrDefault(self.tol)
    
    def setMaxIter(self, value):
        return self._set(max_iter=value)
    
    def getMaxIter(self):
        return self.getOrDefault(self.max_iter)
        
    def _fit(self, dataset):
        
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()
        
        ds_rdd = dataset.select(inputCol).rdd
        
        m = IndexedRowMatrix(ds_rdd)
        
        mu = self.getMu()
        l = self.getL()
        
        if not mu: 
            mu = 1.25 * mat.computeSVD(1).s
        
        print('mu:', mu)
             
        if not l:
            n_cols = mat.numCols()
            n_rows = mat.numRows()
            l = 1.0 / np.sqrt(np.max((n_cols, n_rows))) 
            
        print('l:', l)
        
        pass