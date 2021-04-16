"""
Plot detectors, separately for each color channel.

A detector is a pattern of input-hidden weights that covers a single color channel.

"""

import torch
import yaml
from pathlib import Path

from polyominoworld import configs
from polyominoworld.network import Network
from polyominoworld.params import Params
from polyominoworld.params import param2default
from polyominoworld.figs import plot_hidden_weights_analysis

from ludwig.results import gen_param_paths

SCALE = 1.0  # scale weights so that rounding to nearest integer effectively rounds to nearest mode


param2requests = {

    'colors': [(
        'red',
        'green',
        'blue',
    )],

    'learning_rate': [0.2],
    'num_epochs': [100],
    'hidden_size': [16],


}

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

            # for each hidden weight pattern
            for h_id, hi in enumerate(h_x):

                # get weights to all color channels - separate each when plotting
                hi_single_channel = hi.reshape((3, configs.World.max_x, configs.World.max_y))
                # scale so that modes of hi are aligned with integers in base 1
                hi_single_channel_scaled = hi_single_channel * SCALE

                # plot
                x_tick_labels = [f'x{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[0])]
                y_tick_labels = [f'y{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[1])]
                plot_hidden_weights_analysis(hi_single_channel_scaled,
                                             title=f'detector {h_id:0>2}',
                                             )

                if input('Press Enter to continue'):
                    continue

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
