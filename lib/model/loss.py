import torch 
import torch.nn as nn

from torch.nn.modules.container import Sequential
from torch import Tensor 

def ssw_loss(model: Sequential) -> Tensor: 
    """Description. 
    Compute sum of squared weights using l2 norm."""

    squared = torch.tensor([p.norm(2)**2 for p in model.parameters()]) 

    return torch.sum(squared)

def sse_loss(output: Tensor, target: Tensor) -> Tensor: 
    """Description. Compute sum of squared errors."""

    return torch.sum((output - target)**2)

def mape_loss(output: Tensor, target: Tensor) -> float: 
    """Description. Mean Absolute Percentage Error."""

    if not isinstance(output, Tensor): 
        output = torch.tensor(output)
    if not isinstance(target, Tensor): 
        target = torch.tensor(target)

    return torch.mean(torch.abs(output - target)).item()

class WeightedLoss(nn.Module):
    """Description. 
    Implement custom loss as weighted combination of sum of squared errors (SSE) and sum squared weights (SSW)."""

    def __init__(self, alpha: float, beta: float, optimize: bool=False):
        super(WeightedLoss, self).__init__()
        self.optimize = optimize

        if self.optimize: 
            self.alpha = torch.nn.Parameter(torch.tensor(alpha, requires_grad=True))
            self.beta = torch.nn.Parameter(torch.tensor(beta, requires_grad=True))
        else: 
            self.alpha = torch.tensor(alpha) 
            self.beta = torch.tensor(beta) 

    def forward(self, model: Sequential, output: Tensor, target: Tensor) -> Tensor: 
        """Description. Apply weighted loss function as forward pass."""
                
        ssw = ssw_loss(model)
        sse = sse_loss(output, target)

        if self.optimize: 
            loss = torch.sigmoid(self.alpha) * ssw + torch.sigmoid(self.beta) * sse
        else: 
            loss = self.alpha * ssw + self.beta * sse

        return loss