from model import Model
from typing import List, Tuple
import torch.nn as nn
import torch
from tqdm import tqdm


class PINNTrainer():
    def __init__(
            self,
            input_shape:int, 
            output_shape:int,
            num_layers:int,
            hidden_layer_size:int,
            activation:str,
            output_activation:str,
            loss_fn: str,
            optimizer: str,
            learning_rate: float,
            initial_conditions: List[Tuple[float, float]],
            boundary_conditions: List[Tuple[float, float]],
            loss_weights: List[int],
            data_path: str,
            num_sampled_points: int,
            clip_max_norm: float,
            **kwargs
    ):
        
        self.model = Model(
            input_shape, 
            output_shape,
            num_layers,
            hidden_layer_size,
            activation,
            output_activation
        )   

        self.param_dict = dict()
        for key, value in kwargs.items():
            self.param_dict[key] = value

        self.data_path = data_path
        self.tmax = 0.0
        self.data = self.process_data()

        self.optimizer = getattr(torch.optim, optimizer)(
            list(self.model.parameters()) + [v for v in self.param_dict.values() if v.requires_grad],
            lr=learning_rate
        )
        self.loss_fn = getattr(nn, loss_fn)()

        self.clip_max_norm = clip_max_norm
        self.loss_weights = loss_weights
        assert len(self.loss_weights) == 4, "For the losses not implimented please input 0.0 and make the size of loss weights 4"
        self.num_sampled_points = num_sampled_points
        self.ic = initial_conditions
        self.bc = boundary_conditions
        self.input_shape = input_shape

    def train(self, num_epochs):
        train_loss = []
        x, y = self.data
        self.model.train()
        for epoch in tqdm(range(num_epochs)):
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            ode_loss = self.ode_loss()
            data_loss = self.data_loss(x, y, y_pred, self.loss_fn)
            boundary_loss = self.boundary_loss()
            ic_loss = self.ic_loss()

            loss = data_loss * self.loss_weights[0] + ode_loss * self.loss_weights[1] + boundary_loss * self.loss_weights[2] + ic_loss * self.loss_weights[3]         
            loss.backward()
            
            if self.clip_max_norm:
                pass
            self.optimizer.step()
            train_loss.append(loss.item())
        
        return train_loss

    @staticmethod
    def derivative(y, x):
        return torch.autograd.grad(
                y, x, grad_outputs=torch.ones_like(y), create_graph=True
            )[0]

    def process_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def ode(self, y, x):
        pass

    def initial_condition(self):
        x, y = [], []
        for ic in self.ic:
            x.append(ic[0])
            y.append(ic[1])

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def boundary_conditons(self):
        '''
        TBDL
        '''
        pass
    
    def data_loss(self, x:torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor,loss_fn: nn.Module):
        loss = loss_fn(y_pred, y)
        return loss

    def ode_loss(self):
        x = torch.rand(self.num_sampled_points, self.input_shape) * self.tmax
        x.requires_grad_(True)
        y_pred = self.model(x)

        ode_val = self.ode(y_pred, x)
        return ode_val.pow(2).mean()

    def boundary_loss(self, **kwargs):
        return 0.0
    
    def ic_loss(self, **kwargs):
        x, y = self.initial_condition()
        y_pred = self.model(x)
        loss = (y_pred - y).pow(2).mean()
        return loss
