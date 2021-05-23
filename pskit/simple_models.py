import numpy as np
import pandas as pd
from .base import PipelineLogger, ExperimentBaseClassifier
from sklearn.linear_model import LogisticRegression


class RandomClassifier(ExperimentBaseClassifier, PipelineLogger):
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.class_weights = y.value_counts(normalize=True)
        self.classes_ = self.class_weights.index.to_list()
        self.weights = self.class_weights.to_list()
        return self
        
    def predict(self, X):
        np.random.seed(42)
        num_objects = X.shape[0]
        output = np.random.choice(self.classes_, size=num_objects, p=self.weights)
        return output
    
    def predict_proba(self, X):
        pred_output = self.predict(X)
        proba_output = np.zeros(shape=(X.shape[0], len(self.classes_)))
        for i, v in enumerate(self.classes_):
            proba_output[pred_output==v, i] = 1
        return proba_output

class UnivariateLogisticRegression(LogisticRegression, ExperimentBaseClassifier, PipelineLogger):

    def set_col(self, col_name):
        self.col_name = col_name
        
    def fit(self, X, y):
        X_ = X[[self.col_name]]
        super().fit(X_, y)
        return self
        
    def predict(self, X):
        X_ = X[[self.col_name]]
        
        return super().predict(X_)
        
    def predict_proba(self, X):
        X_ = X[[self.col_name]]
        return super().predict_proba(X_)