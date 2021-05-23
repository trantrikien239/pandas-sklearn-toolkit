######### IMPORT ########
import pickle
from itertools import cycle
from time import time
from tqdm.auto import tqdm
import shutil
from pathlib import Path

# Pandas, Numpy
import pandas as pd
import numpy as np
from numpy import interp
from matplotlib import pyplot as plt
pd.set_option("display.max_columns", None)

# Model evaluation
from sklearn.metrics import plot_confusion_matrix, roc_auc_score,  auc, \
    precision_recall_fscore_support, classification_report, roc_curve, plot_roc_curve

# Sklearn pipeline
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn import set_config
from sklearn.pipeline import Pipeline
set_config(display = 'diagram')

from .base import PipelineLogger

class KFeatureUnion(FeatureUnion):
    def _hstack(self, Xs):
        cols = [X.columns.tolist() for X in Xs]
        dtypes = []
        for X in Xs:
            dtypes.append([str(X[col].dtype) for col in X])
        cols = np.hstack(cols)
        dtypes = np.hstack(dtypes)
        data = pd.DataFrame(super()._hstack(Xs), columns = cols)
        print('====Converting columns types====')
        for col, dtype in tqdm(zip(cols, dtypes)):
            data[col] = data[col].astype(dtype)
        return data

class KColumnTransformer(ColumnTransformer):
    def _hstack(self, Xs):
        cols = [X.columns.tolist() for X in Xs]
        dtypes = []
        for X in Xs:
            dtypes.append([str(X[col].dtype) for col in X])
        cols = np.hstack(cols)
        dtypes = np.hstack(dtypes)
        data = pd.DataFrame(super()._hstack(Xs), columns = cols)
        print('====Converting columns types====')
        for col, dtype in tqdm(zip(cols, dtypes)):
            data[col] = data[col].astype(dtype)
        return data
