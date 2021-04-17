"""
Plot distribution of input-hidden weights to determine modes.
modes are used when discretizing detectors
"""
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
from pathlib import Path


from polyominoworld.network import Network
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests


from ludwig.results import gen_param_paths

SCALE = 0.9  # scale weights so that rounding to nearest integer effectively rounds to nearest mode
SHIFT = 0.1

if __name__ == '__main__':

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            isolated=True,
            runs_path=Path(__file__).parent.parent / 'runs',
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        for path_to_net in list(param_path.rglob('model.pt'))[:1]:

            print(f'Loading net from {path_to_net}')

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()
            h_x = net.h_x.weight.detach().numpy()  # [num hidden, num world cells]

            max_x = 4
            fig, ax = plt.subplots()
            ax.hist(h_x.flatten() * 1, bins=128, range=[-max_x, +max_x], histtype='step')
            ax.hist(h_x.flatten() * SCALE + SHIFT, bins=128, range=[-max_x, +max_x])
            ax.set_xlim([-max_x, max_x])
            x_ticks = np.arange(-max_x, max_x)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks)
            plt.show()

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
