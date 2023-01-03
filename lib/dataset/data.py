import yfinance as yf 

from sklearn.preprocessing import MinMaxScaler

import torch

from torch import Tensor
from typing import List
from dataclasses import dataclass

from .indicators import _add_indicators
from .preprocess import (
    _add_target, 
    _select_features, 
    _train_test_split, 
    _scale,
)

@dataclass 
class Data: 
    """Description. 
    Financial data object to use as input in the Bayesian NN.
    
    Attributes: 
        - ticker: stock ticker
        - start_date: beginning of observation
        - end_date: end of observation 
        - batch_size: size of each training/validation batch
        - train_prop: proportion of training examples
        - ema_n_days: parameter of exponential moving average
        - d_period: parameter for stochastic %D
        - target_name: name of the target variable

    Returns: object of type Data with multiple properties."""

    ticker: str
    start_date: str
    end_date: str 
    batch_size: int
    train_prop: float
    ema_n_days: List
    d_period: int
    target_name: str="Close+1"

    def __post_init__(self): 
        self.df = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        
        self._scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        self._preprocess()
        self._to_batches()

    def _preprocess(self): 
        """Description. Apply preprocessing steps to original data."""
        
        self.df = _add_target(self.df)
        df_train, df_test, self.train_period, self.test_period = _train_test_split(self.df, self.train_prop)

        self._df_train = _add_indicators(df_train, self.ema_n_days, self.d_period)
        self._df_test = _add_indicators(df_test, self.ema_n_days, self.d_period) 

        self._X_train = _select_features(self._df_train, self.target_name)
        self.X_test = _select_features(self._df_test, self.target_name)

        self._X_train = Tensor(_scale(self._X_train.values, self._scaler_X, fit=True)) 
        self.X_test = Tensor(_scale(self.X_test.values, self._scaler_X)) 
        
        self._y_train = self._df_train.loc[:, self.target_name].values.reshape(-1, 1)
        self.y_test = self._df_test.loc[:, self.target_name].values.reshape(-1, 1)

        self._y_train = Tensor(_scale(self._y_train, self.scaler_y, fit=True)) 
        self.y_test = Tensor(_scale(self.y_test, self.scaler_y)) 
         

    def _to_batches(self): 
        """Description. Make batches of length window size out of features and targets."""

        def _make_loader(X: Tensor, y: Tensor) -> List: 
            features, targets = torch.split(X, self.batch_size), torch.split(y, self.batch_size)
            return [(f, t) for f, t in zip(features, targets)]

        self.trainloader = _make_loader(self._X_train, self._y_train)
        self.valloader = self.trainloader[1:-1]