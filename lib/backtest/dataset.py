from lib.dataset.data import Data
from lib.dataset.preprocess import  unscale_tensor
from lib.model.loss import mape_loss 

import pandas as pd 
import numpy as np 

from torch import Tensor

from pandas.core.frame import DataFrame 
from torch.nn.modules.container import Sequential
from typing import (
    Tuple, 
    Optional, 
    Union
) 

def create_backtest_dataset(
    data: Data, 
    model: Optional[Sequential]=None, 
    preds: Optional[Union[np.ndarray, Tensor]]=None, 
    training: bool=False
) -> Tuple: 
    """Description. 
    Return pandas DataFrame with days, actual, predicted values as columns and MAPE.
    
    Attributes: 
        - data: Data object with training and test sets
        - model: optional neural network model 
        - preds: optional vector of predictions 
        - training: training indicator."""
        

    if training: 
        if model != None and preds == None:
            preds = model(data._X_train)
            preds = unscale_tensor(x=preds, scaler=data.scaler_y)
        y = data._y_train
    else:
        if model != None and preds == None:
            preds = model(data.X_test) 
            preds = unscale_tensor(x=preds, scaler=data.scaler_y)
        y = data.y_test

    y = unscale_tensor(x=y, scaler=data.scaler_y)

    n = len(y)
    test_mape = round(100 * mape_loss(preds, y), 2) 
    
    df_backtest = pd.DataFrame(
        data={
            "Days": [i for i in range(n)], 
            "Model": preds.reshape(-1,), 
            "Actual": y.reshape(-1, )
        }
    )

    return df_backtest, test_mape

def generate_price_distribution(data: Data, model: Sequential, n_sim: int=100, training: bool=True) -> DataFrame: 
    """Description. 
    Run bayesian NN multiple times and return a distribution of predicted prices.
    
    Attributes: 
        - data: Data object 
        - n_sim: number of times the model is called
        - training: training indicator."""

    if training: 
        X = data._X_train
        dates = data._df_train.index
    else:
        X = data.X_test
        dates = data._df_test.index 

    res = {
        date.strftime("%Y-%m-%d"): [
            unscale_tensor(x=model(x), scaler=data.scaler_y)[0][0] 
            for _ in range(n_sim)
        ]
        for x, date in zip(X, dates)
    } 

    return pd.DataFrame(res)    