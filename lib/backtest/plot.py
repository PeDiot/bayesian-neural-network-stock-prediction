import seaborn as sns 
import matplotlib.pyplot as plt 

from pandas.core.frame import DataFrame
from typing import Tuple 

def backtest_plot(
    ticker: str, 
    df_backtest: DataFrame, 
    mape: float, 
    period: Tuple,
    plot_dims: Tuple=(17, 6)): 
    """Description. 
    Visualize model"s performance: actual vs predicted prices & mape."""

    start_date, end_date = period

    fig, axes = plt.subplots(ncols=2, figsize=plot_dims)
    fig.suptitle(f"{ticker} closing prices between {start_date} and {end_date} | MAPE={mape}%", size=14)

    sns.lineplot(data=df_backtest, x="Days", y="Model", label="Model", ax=axes[0])
    sns.lineplot(data=df_backtest, x="Days", y="Actual", label="Actual", ax=axes[0])

    axes[0].set_xlim([0, df_backtest.shape[0]])
    axes[0].set_ylabel("$")
    axes[0].set_title("Evolution of actual & predicted values")

    sns.scatterplot(
        data=df_backtest, 
        x="Actual", 
        y="Model", 
        color="black", 
        alpha=.5,
        marker="+", 
        ax=axes[1])

    x_start = df_backtest["Actual"].min()
    x_end = df_backtest["Actual"].max()
    sns.lineplot(
        x=[x_start, x_end], 
        y=[x_start, x_end], 
        color="black", 
        linestyle="dotted", 
        label="y=x")

    axes[1].set_xlim([x_start, x_end])
    axes[1].set_title("Predicted VS actual")

def plot_predicted_distribution(
    predicted_prices: DataFrame, 
    date: str, 
    model_name: str, 
    plot_dims: Tuple=(18, 6)
): 
    """Description. 
    Make KDE plot and boxplot for distribution of predicted prices at a given date."""

    fig, axes = plt.subplots(ncols=2, figsize=plot_dims)

    title = f"Distribution of predicted prices for {date} | Model={model_name}"
    fig.suptitle(title, size=14)

    sns.kdeplot(
        data=predicted_prices, 
        x=date, 
        shade=True, 
        bw_adjust=.25, 
        ax=axes[0])

    sns.boxplot(
        data=predicted_prices, 
        x=date, 
        flierprops={"marker": "x"},
        boxprops={"facecolor": (.4, .6, .8, .5)},
        medianprops={"color": "coral"},
        showmeans=True,
        meanprops={"marker": "+", "markersize": "14"}, 
        ax=axes[1])

    for ax in axes: 
        ax.set_xlabel("Price ($)")
        ax.set_ylabel(" ")
        
    axes[0].set_title("Kernel Density Estimate (KDE) plot")
    axes[1].set_title("Boxplot")