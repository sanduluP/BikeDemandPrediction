import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import random
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


class XBGRegressor:

    def __init__(self):
        self.model = None

    def randomParameterSearch(self, iter=20):
        # parameters for xgboost: random forest train in parallel, xgboost do in sequential
        params = {'max_depth': [3, 5, 6, 10, 15, 20],
                  'learning_rate': [0.01, 0.1, 0.2, 0.3],
                  'subsample': np.arange(0.5, 1.0, 0.1),
                  'colsample_bytree': np.arange(0.4, 1.0, 0.1),
                  'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
                  'n_estimators': [100, 500, 1000]
                  }

        # Using params to search for best hyperparameters
        # Loading the base model
        xgbr = XGBRegressor(seed=20)
        return RandomizedSearchCV(estimator=xgbr,
                                  param_distributions=params,
                                  n_iter=iter,
                                  cv=5,
                                  verbose=1,
                                  random_state=0)

    def trainBestParam(self, best_param):
        xgbr = XGBRegressor(seed=20,
                            n_estimators=best_param.get('n_estimators'),
                            subsample=best_param.get('subsample'),
                            max_depth=best_param.get('max_depth'),
                            learning_rate=best_param.get('learning_rate'),
                            colsample_bytree=best_param.get('colsample_bytree'),
                            colsample_bylevel=best_param.get('colsample_bylevel'))
        return xgbr
