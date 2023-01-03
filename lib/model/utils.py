import numpy as np 

from rich.table import Table

import torch
import torch.nn as nn 

from torch.nn.modules.container import Sequential


def generate_bnn_summary(bnn_model: Sequential) -> Table: 
    """Description. Return bayesian NN summary as rich Table."""
 
    table = Table(title="Bayesian NN architecture", show_lines=True)

    table.add_column("Layer (type)", justify="left")
    table.add_column("Distrbution params", justify="left")
    table.add_column("Tensors", justify="left")
    table.add_column("Param #", justify="left")

    for ix, layer in enumerate(bnn_model): 

        params = layer.__dict__["_parameters"]
        if len(params) == 0: 
            param_names = " "
        else: 
            param_names = " | ".join(list(params.keys()))

        sizes = [p.size() for p in params.values()]
        n_params = np.sum([np.prod(s) for s in sizes]) 

        sizes = " | ".join([str(s) for s in sizes])

        table.add_row(str(layer), param_names, sizes, str(n_params)) 
    
    return table 

def reset_all_weights(model: Sequential) -> None:
    """Description. 
    Reset all weights in a sequential network."""

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(fn=weight_reset)