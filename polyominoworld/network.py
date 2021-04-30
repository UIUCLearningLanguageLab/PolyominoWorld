from typing import List

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

        if params.hidden_activation_function == 'tanh':
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

        # hidden weights
        self.h_xs = torch.nn.ModuleList()
        hs_previous = self.input_size
        for hs in params.hidden_sizes:
            h_x = torch.nn.Linear(hs_previous, hs, bias=True)
            hs_previous = hs
            h_x.weight.data.uniform_(-self.params.weight_init, self.params.weight_init)
            torch.nn.init.zeros_(h_x.bias)
            self.h_xs.append(h_x)

        # output weights
        self.y_h = torch.nn.Linear(hs_previous, self.output_size, bias=True)
        self.y_h.weight.data.uniform_(-self.params.weight_init, self.params.weight_init)
        torch.nn.init.zeros_(self.y_h.bias)

    def forward(self,
                x: torch.tensor,
                ) -> torch.tensor:

        h = x
        for h_x in self.h_xs:
            z_h = h_x(h)
            h = self.hidden_act(z_h)

        z_y = self.y_h(h)

        if self.output_act is not None:
            y = self.output_act(z_y)
        else:
            y = z_y

        return y

    def compute_hs(self,
                   x: torch.tensor,
                   ) -> List[torch.tensor]:
        """compute only the hidden states, for offline analysis"""

        hs = []
        h = x
        for h_x in self.h_xs:
            z_h = h_x(h)
            h = self.hidden_act(z_h)
            hs.append(h)

        return hs
