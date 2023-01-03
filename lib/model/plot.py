import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 

from typing import Dict, Tuple

def training_plot_summary(results: Dict, plot_dims: Tuple=(18, 7)): 
    """Description. 
    Returns a visualisation of weighted loss on training and validation sets and validation MAPE."""

    df_plot = pd.DataFrame(results)
    df_plot.loc[:, "val_mape"] = 100 * df_plot.loc[:, "val_mape"]

    last_mape = round(df_plot.iloc[-1, 3], 2) 

    fig, axes = plt.subplots(ncols=2, figsize=(18, 7))
    fig.suptitle(f"Number of epochs={df_plot.shape[0]} | Validation MAPE = {last_mape}%", size=14)

    sns.lineplot(data=df_plot, x="epoch", y="train_loss", color="blue", ax=axes[0], label="Train")
    sns.lineplot(data=df_plot, x="epoch", y="val_loss", color="orange", ax=axes[0], label="Validation")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Average Weighted Loss")

    sns.lineplot(data=df_plot, x="epoch", y="val_mape", color="purple", ax=axes[1], label="MAPE") 
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("%")
    axes[1].set_title("Average Mean Absolute Percentage Error") 