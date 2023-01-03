from .loss import WeightedLoss, mape_loss 
from lib.dataset import unscale_tensor, Data

from tqdm import tqdm 
import numpy as np 
import torch 

from torch.nn.modules.container import Sequential
from torch import Tensor 

from sklearn.preprocessing._data import MinMaxScaler

from typing import Optional, Tuple

def _train(
    model: Sequential, 
    features: Tensor, 
    target: Tensor, 
    criterion, 
    optimizer) -> float:
    """Description. Train sequential Bayesian NN for a given bacth. 
    
    Attributes: 
        - model: Bayesian NN
        - features: technical indicators 
        - target: next-day closing price
        - optimizer: algorithm used to update weights
        - criterion: loss function 

    Returns: running training loss."""

    model.train()
    
    def closure() -> Tensor: 
        optimizer.zero_grad()
        output = model(features) 
        
        if isinstance(criterion, WeightedLoss): 
            loss = criterion(model, output, target)   
        else: 
            loss = criterion(output, target)

        loss.backward()     
        return loss

    loss = closure()
    optimizer.step(closure) 

    return loss.item() 

def _validation(
    model: Sequential, 
    features: Tensor, 
    target: Tensor, 
    criterion, 
    scaler: Optional[MinMaxScaler]=None) -> Tuple:
    """Description. Test sequential Bayesian NN on validation data. 
    
    Attributes: 
        - model: Bayesian NN
        - valloader: list of validation batches with features and closing prices
        - criterion: loss function 
        - epoch: running epoch
        
    Returns: 
        - running validation loss
        - validation MAPE."""

    with torch.no_grad():
        model.eval()
        
        output = model(features)

        if isinstance(criterion, WeightedLoss):
            running_loss = criterion(model, output, target)
        else: 
            running_loss = criterion(output, target)
        
        if scaler != None: 
            output = unscale_tensor(output, scaler)
            target = unscale_tensor(target, scaler)
            
        mape = mape_loss(output, target)            
            
        return running_loss.item(), mape

def run_one_epoch(
    model: Sequential, 
    data: Data, 
    criterion, 
    optimizer, 
    epoch: int, 
    num_epochs: int, 
) -> Tuple: 
    """Description. 
    Model training and validation over one epoch. Update results.
    
    Attributes: 
        - model: neural network to optimize
        - data: Data object with trainloader and valloader
        - optimizer: algorithm used to update weights
        - criterion: loss function 
        - epoch: running epoch 
        - num_epochs: total number of epochs
        - results: dictionnary to store loss values
        
    Returns: 
        - training loss
        - validaiton loss
        - validation mape."""

    train_losses, val_losses, mape_values = [], [], []
    n_batches = len(data.trainloader)

    loop = tqdm(range(n_batches))
    loop.set_description(f"Epoch [{epoch + 1}/{num_epochs}]")

    for ix in loop: 

        X_tr, y_tr = data.trainloader[ix]
        train_loss = _train(model, X_tr, y_tr, criterion, optimizer)

        train_losses.append(train_loss)

        if ix < len(data.valloader)-1: 
            X_val, y_val = data.valloader[ix+1]
            val_loss, mape = _validation(model, X_val, y_val, criterion, scaler=data.scaler_y)

            val_losses.append(val_loss)
            mape_values.append(mape)

            loop.set_postfix(train_loss=train_loss, val_loss=val_loss, val_mape=mape) 

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_loss) 
    val_mape = np.mean(mape_values) 

    return train_loss, val_loss, val_mape 