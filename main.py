import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from core import PINNTrainer
from typing import Tuple, List

np.random.seed(69)
torch.manual_seed(69)

class Trianer(PINNTrainer):
    def __init__(
        self,
        input_shape:int = 1, 
        output_shape:int = 3,
        num_layers:int = 3,
        hidden_layer_size:int = 256,
        activation:str = 'ReLU',
        output_activation:str = 'Sigmoid',
        loss_fn: str = 'MSELoss',
        optimizer: str = 'Adam',
        learning_rate: float = 1e-4,
        initial_conditions: List[Tuple[float, float]] = [[[0], [1, 0, 0]]],
        boundary_conditions: List[Tuple[float, float]] = None,
        loss_weights: List[int] = [5, 1, 0, 5],
        data_path: str = 'reaction_data.csv',
        num_sampled_points: int = 100,
        clip_max_norm: float = 1.0,
        **kwargs                
    ):
        super().__init__(
            input_shape,
            output_shape,
            num_layers,
            hidden_layer_size,
            activation,
            output_activation,
            loss_fn,
            optimizer,
            learning_rate,
            initial_conditions,
            boundary_conditions,
            loss_weights,
            data_path,
            num_sampled_points,
            clip_max_norm,
            **kwargs
        )

    def ode(self, y, x):
        grads = []
        n = self.param_dict.get('n')
        k = self.param_dict.get('k')
        us = self.param_dict.get('us')
        v = self.param_dict.get('v')


        for i in range(y.shape[-1]):
            f_grad = self.derivative(y[:, i], x)
            grads.append(f_grad)

        grads = torch.cat(grads,dim=-1)
        r = []
        for i in range(y.shape[-1]):
            r.append(
                (((y[:, i].reshape(-1, 1)) ** n[i]) * k[:, 0]).unsqueeze(1)
            )
        
        r = torch.cat(r, dim=1)
        rate = torch.diagonal(r @ v, dim1=-2, dim2=-1) / us
        
        return grads - rate # = 0
    
    def process_data(self):
        reactant_data = pd.read_csv(self.data_path)
        time_column = reactant_data.columns[0]  # First column name
        concentration_columns = reactant_data.columns[1:]  # All other columns
        
        x = np.array([reactant_data[time_column].values])
        y = np.array(reactant_data[concentration_columns].values)
        
        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32)

        self.tmax = torch.max(x[:,0]).item() #Tmax
        return x, y

    def eval(self):
        reactant_data = pd.read_csv('reaction_data.csv')
        time_column = reactant_data.columns[0]  # First column name
        concentration_columns = reactant_data.columns[1:]  # All other columns

        x = np.array([reactant_data[time_column].values])
        y = np.array(reactant_data[concentration_columns].values)

        x = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        y = torch.tensor(y, dtype=torch.float32)

        y_pred = self.model(x)

        x = x.reshape(-1,).numpy()
        y = y.numpy()
        y_pred = y_pred.detach().numpy()

        plt.figure()
        plt.plot(x, y_pred[:, 0], label='Pred')
        plt.plot(x, y[:, 0], label='True')
        plt.legend()

        plt.figure()
        plt.plot(x, y_pred[:, 1], label='Pred')
        plt.plot(x, y[:, 1], label='True')
        plt.legend()

        plt.figure()
        plt.plot(x, y_pred[:, 2], label='Pred')
        plt.plot(x, y[:, 2], label='True')
        plt.legend()

        plt.show()



trainer = Trianer(
    k=torch.tensor([
    [2],
    [0.02], 
    [0.1], 
    [0.5]
    ], dtype=torch.float32, requires_grad=True),
n=torch.tensor([
    [1.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], requires_grad=True, dtype=torch.float32),
v=torch.tensor([
[-1.0, 1.0, 0.0],
[-1.0, 0.0, 1.0],
[1.0, -1.0, 0.0],
[1.0, 0.0, -1.0]
], dtype=torch.float32),
us=torch.tensor(1.02, dtype=torch.float32),
)
trainer.train(100)
print(trainer.param_dict['v'])
trainer.eval()