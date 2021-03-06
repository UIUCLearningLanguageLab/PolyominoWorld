"""
Analyse hidden states using accuracy of a linear readout at hidden layer.
Perfect accuracy indicates linear separability.

implemented with the moore-penrose pseudo-inverse:
least squares approximation of linear transformation = W  = LP+
where L has binary class labels in cols, and P+ is pseudo-inv of representations in columns

see: https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full


"""
import torch
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np

from polyominoworld.evaluate import evaluate_linear_readout
from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.figs import plot_lines
from polyominoworld import configs

from ludwig.results import gen_param_paths

FEATURE_TYPE = 'shape'
HIDDEN_LAYER_ID = 1

if __name__ == '__main__':

    configs.Device.gpu = False

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            # isolated=True,
            # runs_path=Path(__file__).parent.parent / 'runs',
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # use all locations, rotations, shapes, and colors - and filter later
        world = World(params)
        data = DataSet(world.generate_sequences(leftout_colors=params.leftout_colors,
                                                leftout_shapes=params.leftout_shapes,
                                                leftout_variants=params.leftout_variants,
                                                leftout_positions=get_leftout_positions(params.leftout_half),
                                                ),
                       params,
                       name='re-generated')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        paths_to_net = list(sorted(param_path.rglob(f'model_*.pt')))
        if not paths_to_net:
            raise FileNotFoundError('Did not find any model files')

        rep_names = [p.name for p in param_path.glob('*') if p.is_dir()]

        net = Network(params)

        x_tick2ys = defaultdict(list)
        for path_to_net in paths_to_net:

            print(f'Loading net from {path_to_net}')

            # load weights into net
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.requires_grad_(False)
            net.eval()

            accuracy = evaluate_linear_readout(data, net, FEATURE_TYPE, HIDDEN_LAYER_ID)
            print(f'accuracy of readout={accuracy:.4f}')
            print()

            x_tick = int(path_to_net.stem.lstrip('model_'))
            x_tick2ys[x_tick].append(accuracy)

        # compute  baselines
        baseline_acc = evaluate_linear_readout(data, net, FEATURE_TYPE, HIDDEN_LAYER_ID, state_is_input=True)
        random_acc = evaluate_linear_readout(data, net, FEATURE_TYPE, HIDDEN_LAYER_ID, state_is_random=True)

        # plot
        ys = []
        for x_tick, y in sorted(x_tick2ys.items()):
            ys.append(y)
        ys = np.array(ys).T
        x_ticks = list(sorted(x_tick2ys))
        plot_lines(
            ys,
            title=f'{param_path.name}\nLinear readout at hidden layer with id={HIDDEN_LAYER_ID}\n',
            x_axis_label='Training Step',
            y_axis_label=f'{FEATURE_TYPE.capitalize()} Accuracy',
            x_ticks=x_ticks,
            labels=rep_names,
            y_lims=[0, 1],
            baseline_input=baseline_acc,
            baseline_random=random_acc,
            label_last_x_tick_only=True,
        )

        break  # do not keep searching for models - regular pattern ids are defined for first model only
