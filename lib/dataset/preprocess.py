import numpy as np 
from torch import Tensor

from sklearn.preprocessing._data import MinMaxScaler
from pandas.core.frame import DataFrame
from typing import Tuple 

def _add_target(df: DataFrame) -> DataFrame: 
    """Description. Add t+1 closing prices."""

    df.loc[:, "Close+1"] = df["Close"].shift(1)
    return df 

def _train_test_split(df: DataFrame, train_prop: float, target_name: str="Close+1") -> Tuple: 
    """Description. 
    Split data into training / test sets and return training / test period."""

    n = df.shape[0]
    train_idxs = [i for i in range(int(train_prop * n))]

    df_train = df.iloc[train_idxs, :]
    df_test = df.drop(df.iloc[train_idxs, :].index.tolist())

    train_period = (df_train.index.min().strftime("%Y-%m-%d"), df_train.index.max().strftime("%Y-%m-%d"))
    test_period = (df_test.index.min().strftime("%Y-%m-%d"), df_test.index.max().strftime("%Y-%m-%d"))

    return df_train, df_test, train_period, test_period

def _select_features(df: DataFrame, target_name: str) -> DataFrame: 
    """Description. Select Open, Low, High and technical indicators."""

    to_remove = ["Close", "Adj Close", "Volume", target_name]
    return df.drop(labels=to_remove, axis=1)   

def _scale(X: np.ndarray, scaler: MinMaxScaler, fit: bool=False) -> np.ndarray: 
    """Description. Scaled numpy array between -1 and 1 and return Tensor."""
    if fit:
      scaler.fit(X)

    X_scaled = scaler.transform(X)
    return X_scaled

def unscale_tensor(x: Tensor, scaler: MinMaxScaler) -> np.ndarray: 
    """Description. Return unscaled numpy array."""

    return scaler.inverse_transform(x.detach().numpy())