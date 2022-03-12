#TODO Add test's
# Import general modules used e.g. for plotting.
import logging
import pandas as pd

from typing import Tuple

import torch
from torch.nn import Module
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

def train_epoch(network: Module,
                optimiser: AdamW,
                data_loader: DataLoader,
                device:torch.device
               ) -> Tuple[float, float]:
    network.train()
    losses, accuracies = [], []
    for sample_data in data_loader:
        print(type(sample_data))
        data, target = sample_data[r'data'], sample_data[r'target']
        data, target = data.to(device=device), target.to(device=device)

        # Process data by Hopfield-based network.
        model_output = network.forward(input=data)

        # Update network parameters.
        optimiser.zero_grad()
        loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')
        loss.backward()
        clip_grad_norm_(parameters=network.parameters(), max_norm=1.0, norm_type=2)
        optimiser.step()

        # Compute performance measures of current model.
        accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
        accuracies.append(accuracy.detach().item())
        losses.append(loss.detach().item())
    
    # Report progress of training procedure.
    return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def eval_iter(network: Module,
              data_loader: DataLoader,
              device:torch.device
             ) -> Tuple[float, float]:
 
    network.eval()
    with torch.no_grad():
        losses, accuracies = [], []
        for sample_data in data_loader:
            data, target = sample_data[r'data'], sample_data[r'target']
            data, target = data.to(device=device), target.to(device=device)

            # Process data by Hopfield-based network.
            model_output = network.forward(input=data)
            loss = binary_cross_entropy_with_logits(input=model_output, target=target, reduction=r'mean')

            # Compute performance measures of current model.
            accuracy = (model_output.sigmoid().round() == target).to(dtype=torch.float32).mean()
            accuracies.append(accuracy.detach().item())
            losses.append(loss.detach().item())

        # Report progress of validation procedure.
        return (sum(losses) / len(losses), sum(accuracies) / len(accuracies))


def operate(network: Module,
            optimiser: AdamW,
            data_loader_train: DataLoader,
            data_loader_eval: DataLoader,
            device:torch.device,
            num_epochs: int = 1,
           ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    losses, accuracies = {r'train': [], r'eval': []}, {r'train': [], r'eval': []}
    for epoch in range(num_epochs):
        
        # Train network.
        performance = train_epoch(network, optimiser, data_loader_train, device)
        losses[r'train'].append(performance[0])
        accuracies[r'train'].append(performance[1])
        
        # Evaluate current model.
        performance = eval_iter(network=network,
                                data_loader=data_loader_eval, 
                                device=device)
        if epoch % 5 == 0:
            print("---------------------------------")
            print("epoch:", epoch, "of", num_epochs, "\naccuracy:", performance[1], "\nloss:", performance[0])
        losses[r'eval'].append(performance[0])
        accuracies[r'eval'].append(performance[1])
    
    # Report progress of training and validation procedures.
    return pd.DataFrame(losses), pd.DataFrame(accuracies)