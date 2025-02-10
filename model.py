import torch.nn as nn

class Model(nn.Module):
    def __init__(
            self,
            input_shape, 
            output_shape,
            num_layers,
            hidden_layer_size,
            activation,
            output_activation,
    ):
        super().__init__()

        tmp = []
        for i in range(num_layers):
            if i == 0:
                tmp.append(nn.Linear(input_shape, hidden_layer_size))
                act = getattr(nn, output_activation)() if num_layers == 1 else getattr(nn, activation)()
                tmp.append(act)
            elif i == num_layers-1:
                tmp.append(nn.Linear(hidden_layer_size, output_shape))
                act = getattr(nn, output_activation)()
                tmp.append(act)
            else:
                tmp.append(nn.Linear(hidden_layer_size, hidden_layer_size))
                act = getattr(nn, activation)()
                tmp.append(act)

        self.net = nn.Sequential(
            *tmp
        )

    def forward(self, x):
        return self.net(x)
