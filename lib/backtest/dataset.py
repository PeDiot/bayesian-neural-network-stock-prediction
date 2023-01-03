from lib.dataset.data import Data
from lib.dataset.preprocess import  unscale_tensor
from lib.model.loss import mape_loss 

import pandas as pd 

from torch.nn.modules.container import Sequential

from typing import Tuple

def create_backtest_dataset(model: Sequential, data: Data, training: bool=False) -> Tuple: 
    """Description. 
    Return pandas DataFrame with days, actual, predicted values as columns and MAPE.
    
    Attributes: 
        - model: neural network 
        - data: Data object with training and test sets
        - training: training indicator."""

    if training: 
        preds = model(data._X_train)
        y = data._y_train
    else:
        preds = model(data.X_test) 
        y = data.y_test

    preds = unscale_tensor(x=preds, scaler=data.scaler_y)
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