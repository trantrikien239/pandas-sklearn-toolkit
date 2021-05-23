import numpy as np
import pandas as pd

from .base import ExperimentBaseClassifier, PipelineLogger
from catboost import CatBoostClassifier

class KCatBoostClassifier(CatBoostClassifier, ExperimentBaseClassifier, PipelineLogger):
    """
    Implement custom features compare to Catboost's standard classifier:
    - Automatic split X_eval, y_eval out of X_train, y_train
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def fit(self, X, y, self_evaluate=True, eval_set=None, **kwargs):
        self.log_start()
        if self._init_params.get('cat_features') is not None:
            cat_features_ = [c for c in self._init_params['cat_features'] if c in X.columns]
            self._init_params['cat_features'] = cat_features_
        else:
            cat_features_ = None
        if self._init_params.get('text_features') is not None:
            text_features_ = [c for c in self._init_params['text_features'] if c in X.columns]
            self._init_params['text_features'] = text_features_
        else:
            text_features_ = None
        
        if eval_set is not None:
            X_e, y_e = eval_set
            X_t = X
            y_t = y
        else:
            X_t, X_e, y_t, y_e = self.train_eval_split(X, y, cat_features_, text_features_)

        super().fit(X_t, y_t, eval_set=(X_e, y_e), cat_features=cat_features_, text_features=text_features_)
        if self_evaluate:
            _ = self.evaluate(X_e, y_e)
        self.log_finish()
        return self
        
    def add_na(self, X, y, na_label=0, cat_features_=None, text_features_=None):
        X = pd.concat([X, pd.DataFrame([[np.nan] * X.shape[1]], columns=X.columns)], ignore_index=True)
        y = pd.concat([y, pd.Series([na_label])], ignore_index=True)
        
        if cat_features_ is not None:
            cat_features_ = [c for c in cat_features_ if c in X.columns]
            X[cat_features_] = X[cat_features_].fillna('unk')
        if text_features_ is not None:
            text_features_ = [c for c in text_features_ if c in X.columns]
            X[text_features_] = X[text_features_].fillna('unk')
        return X, y

    def train_eval_split(self, X, y, eval_frac=0.1):
        X_e = X.sample(frac=eval_frac, random_state=42)
        y_e = y.loc[X_e.index]
        X_t = X.drop(X_e.index)
        y_t = y.loc[X_t.index]
        return X_t, X_e, y_t, y_e

        