from ta.momentum import rsi, williams_r, stoch

from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import List, Tuple 


def _calc_ema(close: Series, n_days: int) -> Tuple: 
    """Description. 
    Return exponential moving average and variable name."""
    
    var_name = f"EMA{n_days}"
    ema = close.ewm(span=n_days).mean()

    return var_name, ema

def _add_indicators(df: DataFrame, ema_n_days: List, d_period: int) -> DataFrame: 
    """Description. 
    Add multiple financial indicators: EMA, RSI, William"s R, Stochastic K%, Stochastic D%. 

    Attributes: 
        - df: initial data set
        - ema_n_days: window size for exponential moving average
        - d_period: window size for stochastic D%.

    Returns: transformed data set.
    """

    for n in ema_n_days: 
        ema_var_name, emas = _calc_ema(df["Close"], n)
        df.loc[:, ema_var_name] = emas

    df.loc[:, "RSI"] = rsi(df["Close"])

    df.loc[:, "WILLIAMS_R"] = williams_r(df["High"], df["Low"], df["Close"])

    df.loc[:, "%K"] = stoch(df["High"], df["Low"], df["Close"])
    df.loc[:, "%D"] = df["%K"].rolling(d_period).mean()

    return df.dropna(axis=0)