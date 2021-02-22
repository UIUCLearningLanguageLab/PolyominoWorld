from typing import Any

import torch

from polyominoworld.params import Params
from polyominoworld.helpers import FeatureVector, WorldVector


class Network(torch.nn.Module):
    """network with hidden layer"""

    def __init__(self,
                 params: Params,
                 ) -> None:
        super(Network, self).__init__()

        self.params = params
        self.has_hidden_layer = True

        if params.hidden_activation_function is None:  # use network without hidden layer
            print('Making network without hidden layer')
            self.has_hidden_layer = False

        elif params.hidden_activation_function == 'tanh':
            self.hidden_act = torch.nn.Tanh()
        elif params.hidden_activation_function == 'sigmoid':
            self.hidden_act = torch.nn.Sigmoid()
        elif params.hidden_activation_function == 'relu':
            self.hidden_act = torch.nn.ReLU()
        else:
            raise RuntimeError(f"Hidden activation function {params.hidden_activation_function} not recognized")

        if params.x_type == 'world':
            self.input_size = WorldVector.calc_size()
        elif params.x_type == 'hidden':
            self.input_size = NotImplemented  # TODO
        else:
            raise AttributeError(f'x_type "{params.x_type}" not recognized')

        if params.y_type == 'world':
            self.output_act = torch.nn.Tanh()
            self.output_size = WorldVector.calc_size()
        elif params.y_type == 'features':
            self.output_act = None  # bce loss uses logits
            self.output_size = FeatureVector.calc_size()
        else:
            raise AttributeError(f'y_type "{params.y_type}" not recognized')

        # weights for multi-layer net
        self.h_x = torch.nn.Linear(self.input_size, self.params.hidden_size)
        self.y_h = torch.nn.Linear(self.params.hidden_size, self.output_size)
        # weights for single-layer net
        self.y_x = torch.nn.Linear(self.input_size, self.output_size)
        self.init_weights()

    def init_weights(self):
        self.h_x.weight.data.uniform_(-self.params.weight_init, self.params.weight_init)
        self.h_x.bias.data.uniform_(-self.params.weight_init, self.params.weight_init)

        self.y_h.weight.data.uniform_(-self.params.weight_init, self.params.weight_init)
        self.y_h.bias.data.uniform_(-self.params.weight_init, self.params.weight_init)

        self.y_x.weight.data.uniform_(-self.params.weight_init, self.params.weight_init)
        self.y_x.bias.data.uniform_(-self.params.weight_init, self.params.weight_init)

    def forward(self, x):
        if self.has_hidden_layer:
            z_h = self.h_x(x)
            h = self.hidden_act(z_h)
            z_o = self.y_h(h)
        else:
            z_o = self.y_x(x)

        if self.output_act is not None:
            o = self.output_act(z_o)
        else:
            o = z_o

        return o
