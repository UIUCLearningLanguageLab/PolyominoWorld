"""
Analyse hidden states using accuracy of a linear readout at hidden layer.
Perfect accuracy indicates linear separability.

implemented with the moore-penrose pseudo-inverse:
least squares approximation of linear transformation = W  = LP+
where L has binary class labels in cols, and P+ is pseudo-inv of representations in columns

see: https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00515/full


"""
from typing import Tuple
import torch
import yaml
from numpy.linalg import pinv
from pathlib import Path
from collections import defaultdict
import numpy as np

from polyominoworld import configs
from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.figs import plot_line

from ludwig.results import gen_param_paths

FEATURE_TYPE = 'shape'
USE_NON_LINEARITY = True


def make_l_and_p(data: DataSet,
                 net: Network,
                 use_non_linearity: bool = USE_NON_LINEARITY,
                 state_is_random: bool = False,
                 state_is_input: bool = False,
                 ) -> Tuple[np.array, np.array]:

    h_x = net.h_x.weight.detach().numpy()  # [num hidden, num world cells]

    lis = []
    pis = []
    for event in data.get_events():

        # L has columns of localist target vectors
        if FEATURE_TYPE == 'shape':
            li = [1 if event.shape == s else 0 for s in configs.World.master_shapes]
        elif FEATURE_TYPE == 'color':
            li = [1 if event.color == s else 0 for s in configs.World.master_colors]
        elif FEATURE_TYPE == 'size':
            li = [1 if event.size == s else 0 for s in configs.World.master_sizes]
        else:
            raise AttributeError('Invalid feature type')
        lis.append(li)

        # P has columns of representation vectors
        x = event.world_vector.vector

        # compute baseline state
        if state_is_random:
            state = np.random.permutation(net.forward(x).detach().numpy())
        elif state_is_input:  # yields same results as random_state = True
            state = x.numpy()

        # compute actual state
        else:
            if use_non_linearity:
                state = net.forward(x).detach().numpy()
            else:
                state = h_x @ x.numpy()

        pi = state.T
        pis.append(pi)

    L = np.array(lis).T
    P = np.array(pis).T

    return L, P


def calc_accuracy_of_readout(l_matrix_correct: np.array,
                             l_matrix_predicted: np.array,
                             ) -> float:
    num_correct = 0
    for li_correct, li_predicted in zip(l_matrix_correct.T, l_matrix_predicted.T):
        assert np.sum(li_correct) == 1.0
        id_correct = np.argmax(li_correct)
        id_predicted = np.argmax(li_predicted)
        if id_correct == id_predicted:
            num_correct += 1
    return num_correct / len(l_matrix_correct.T)


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

        # use all locations, rotations, shapes, and colors - and filter later
        world = World(params)
        data = DataSet(world.generate_sequences(leftout_colors=('', ),
                                                leftout_shapes=('', ),
                                                leftout_variants='',
                                                leftout_positions=get_leftout_positions(''),
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
            net.eval()

            # make L and P
            L, P = make_l_and_p(data, net, state_is_random=False)
            # compute W
            W = L @ pinv(P)
            # compute linear readout L
            L_predicted = W @ P  # [num features, num instances]

            # compute accuracy
            accuracy = calc_accuracy_of_readout(L, L_predicted)
            print(f'accuracy of readout={accuracy:.4f}')
            print()

            x_tick = int(path_to_net.stem.lstrip('model_'))
            x_tick2ys[x_tick].append(accuracy)

        # compute baseline based on linear separability of input
        L, P = make_l_and_p(data, net, state_is_input=True)
        W = L @ pinv(P)
        L_predicted = W @ P  # [num features, num instances]
        baseline_input = calc_accuracy_of_readout(L, L_predicted)

        # compute random guessing baseline
        L, P = make_l_and_p(data, net, state_is_random=True)
        W = L @ pinv(P)
        L_predicted = W @ P  # [num features, num instances]
        baseline_random = calc_accuracy_of_readout(L, L_predicted)

        # plot
        ys = []
        for x_tick, y in sorted(x_tick2ys.items()):
            ys.append(y)
        ys = np.array(ys).T
        x_ticks = list(sorted(x_tick2ys))
        plot_line(
            ys,
            title=f'{param_path.name}\nLinear readout at hidden state\nuse non-linearity={USE_NON_LINEARITY}',
            x_axis_label='Epoch',
            y_axis_label=f'{FEATURE_TYPE.capitalize()} Accuracy',
            x_ticks=x_ticks,
            labels=rep_names,
            y_lims=[0, 1],
            baseline_input=baseline_input,
            baseline_random=baseline_random,
        )

        break  # do not keep searching for models - regular pattern ids are defined for first model only