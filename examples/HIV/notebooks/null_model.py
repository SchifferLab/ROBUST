import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import statsmodels.api as sm

class NullModel(TransformerMixin, BaseEstimator):
    """
    Select features where the relative likelihood of the null model is <p.
    The relative likelihood is determiend by comparing the AIC of the two models.
    
    The indices of the features included in Model and Nullmodel are specified in a dictionary:
    
    tests = {index_model: index_null, ... }
    
    Features can only be part of the model or the null_model but not both.
    
    Each (non-null)feature can only be included in one test
    
    """
    def __init__(self,tests=None, p=0.05, drop_null=True):

        if tests is not None:
            self.X_null  = set()
            self.X_model = set()
            for k, v in tests.items():
                model_intersection = self.X_model.intersection(k)
                if model_intersection:
                    raise RuntimeError('Features: '+' '.join(list(map(str, model_intersection)))+ 
                                  ' included in multiple tests')
                if isinstance(k, str):
                    self.X_model.add(k)
                else:
                    self.X_model.update(k)
                if isinstance(v, str):
                    self.X_null.add(v)
                else:
                    self.X_null.update(v)
            if self.X_null.intersection(self.X_model):
                raise RuntimeError('Features: '+' '.join(list(map(str, self.X_null.intersection(self.X_model))))+ 
                                  ' included in model and null model')
        else:
            warnings.warn(RuntimeWarning, 'No tests specified')
        self.tests = tests
        self.p = p
        self.support = None
        self.drop_null = drop_null
        self.models = {}
        
    def fit(self, X, y):
        
        if self.tests is None:
            return self
        
        if isinstance(X, pd.DataFrame):
            self.support = pd.Series(np.zeros(X.shape[1] , dtype=float), index=X.columns)
        else:
            self.support = pd.Series(np.zeros(X.shape[1] , dtype=float), index=np.arange(len(X)))
        
        if self.drop_null:
            self.support[self.X_null] = 1.
        
        for index, index_null in self.tests.items():
            x = X[index]
            x_null = X[index_null]
            f = sm.OLS(y, sm.add_constant(x)).fit()
            f_null = sm.OLS(y, sm.add_constant(x_null)).fit()
            if f_null.aic < f.aic:
                self.support[index] = 1.
            else:
                self.support[index] = np.exp((f.aic-f_null.aic)/2)
                self.models[index] = [f, f_null]
        return self
    
    def transform(self, X):
        return X[self.support.index[self.support<self.p]]