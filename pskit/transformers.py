######### IMPORT ########
from tqdm.auto import tqdm

# Pandas, Numpy
import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)


# Sklearn pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from sklearn.impute import SimpleImputer

from .base import PipelineLogger


class KColumnSelector(BaseEstimator, TransformerMixin, PipelineLogger):
    def __init__(self, columns=None):
        super().__init__()
        self.columns = columns
        if columns is None:
            print('columns is None ===> Select all columns of the fit data')
    def fit(self, X, y=None):
        if self.columns is None:
            self.selected_columns = X.columns.to_list()
        else:
            self.selected_columns = [c for c in self.columns if c in X.columns]
            print(f"""Number of instructed columns: {len(self.columns)}\nNumber of columns in X: {len(X.columns)}\nNumber of selected colums (overlap): {len(self.selected_columns)}""")
        return self
    def transform(self, X):
        self.log_start()
        X_ = X[self.selected_columns]
        self.log_finish()
        return X_

class KSimpleImputer(SimpleImputer):
    def fit(self, X, y=None):
        self._cols = X.columns.tolist()
        self._dtypes = [str(X[col].dtype) for col in X.columns]
        super().fit(X, y)
        return self
        
    def transform(self, X):
        X_ = super().transform(X)
        data = pd.DataFrame(X_, columns = self._cols)
        for col, dtype in tqdm(zip(self._cols, self._dtypes)):
            data[col] = data[col].astype(dtype)
        return data

class KBasicPreprocessor(BaseEstimator, TransformerMixin, PipelineLogger):
    def __init__(self, 
                 lower_cols=True,
                text_features=None,
                lower_text=True,
                cardinality_threshold=0.5, 
                na_threashold=0.998):
        super().__init__()
        self.lower_cols = lower_cols
        self.text_features= text_features
        self.cardinality_threshold = cardinality_threshold
        self.na_threashold = na_threashold
        self.lower_text = lower_text
    
    def lower_column_name(self):
        if self.lower_cols:
            self.X_.columns = [c.lower() for c in self.X_.columns]
    
    def remove_non_feature_cols(self):
        print(self.X_.shape)
        if self.non_feature_cols is not None:
            self.X_ = self.X_[[c for c in self.X_.columns if c not in self.non_feature_cols]]
    
    def remove_high_cardinality(self):
        print(self.X_.shape)
        print("---- Removing high cardinality ----")
        if self.text_features is not None:
            cat_features = [c for c in self.column_type['text'] if c not in self.text_features]
        else:
            cat_features = [c for c in self.column_type['text']]
        cardinality_pct = self.X_[cat_features].nunique() / self.X_[cat_features].notna().sum(axis=0)
        high_cardinality = cardinality_pct[cardinality_pct > self.cardinality_threshold].copy()
        print(f"Features to remove: \n{high_cardinality.to_dict()}")
        self.X_.drop(columns=[c for c in high_cardinality.index.to_list()],
                     inplace=True)
        print("---- Done removing high cardinality ----")
        
    def get_column_type(self, X):
        column_type = {}
        column_type['int'] = X.select_dtypes(include=[c for c in X.dtypes if 'int' in str(c)]).columns.to_list()
        column_type['float'] = X.select_dtypes(
            include=[c for c in X.dtypes if 'float' in str(c)]).columns.to_list()
        column_type['text'] = X.select_dtypes(
            include=[c for c in X.dtypes if 'object' in str(c) or "'O'" in str(c)]).columns.to_list()
        column_type['time'] = X.select_dtypes(include=[c for c in X.dtypes if 'time' in str(c)]).columns.to_list()
        return column_type
    
    def remove_high_na(self):
        print(self.X_.shape)
        print("---- Removing high na ----")
        num_records = self.X_.shape[0]
        na_pct = self.X_.isna().sum(axis=0) / num_records
        high_na = na_pct[na_pct > self.na_threashold].copy()
        print(high_na)
        print(f"Features to remove: \n{high_na.to_dict()}")
        self.X_.drop(columns=high_na.index.to_list(), inplace=True)
        print("---- Done removing high na ----")
    
    def lower_txt(self):
        def lower_text_func(x):
            try:
                return x.lower()
            except:
                return x
        for c in self.column_type['text']:
            self.X_[c] = self.X_[c].apply(lambda x: lower_text_func(x))
        
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        self.log_start()
        self.X_ = X.copy()
        self.lower_column_name()
        self.column_type = self.get_column_type(self.X_)
        self.lower_txt()
        self.remove_high_cardinality()
        self.remove_high_na()
        self.log_finish()
        return self.X_
    
class KPassThroughExcept(BaseEstimator, TransformerMixin, PipelineLogger):
    """
    Applied as an alternative to `remainder='pass_through'` in PSKitColumnTransformer for 
    PSKitFeatureUnion.
    """
    def __init__(self, col_except_func):
        super().__init__()
        self.col_except_func = col_except_func
        
    def fit(self, X, y=None):
        self.except_cols = self.col_except_func(X)
        return self
    
    def transform(self, X):
        self.log_start()
        X_ = X[[c for c in X.columns if c not in self.except_cols]]
        self.log_finish()
        return X_