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
from itertools import combinations
import multiprocessing as mp
from pathlib import Path
import numpy as np
from collections import defaultdict

from polyominoworld.utils import get_leftout_positions, evaluate_detector_combo
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.figs import plot_line
from polyominoworld.evaluate import evaluate_linear_readout

from ludwig.results import gen_param_paths

MIN_COMBO_SIZE = 1
NUM_WORKERS = 6
FEATURE_TYPE = 'size'

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
        paths_to_net = list(sorted(param_path.rglob(f'model.pt')))
        if not paths_to_net:
            raise FileNotFoundError('Did not find any model files')

        rep_names = [p.name for p in param_path.glob('*') if p.is_dir()]

        net = Network(params)

        assert MIN_COMBO_SIZE <= params.hidden_size

        x_tick2ys = defaultdict(list)
        for path_to_net in paths_to_net:

            print(f'Loading net from {path_to_net}')

            # load weights into net
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.requires_grad_(False)
            net.eval()

            # search all combinations of hidden units
            for combo_size in range(MIN_COMBO_SIZE, params.hidden_size + 1):

                # set up parallel processes that read from queue and save results in shared memory + shared memory
                q = mp.Queue(maxsize=NUM_WORKERS)
                score_max = mp.Value('d')
                score_max.value = +0.0
                lock = mp.Lock()
                pool = mp.Pool(NUM_WORKERS,
                               initializer=evaluate_detector_combo,  # TODO speed up using batching
                               initargs=(q,
                                         data,
                                         net,
                                         FEATURE_TYPE,
                                         score_max,
                                         lock,
                                         ))

                print(f'Searching combo size={combo_size}')
                for h_ids in combinations(range(params.hidden_size), combo_size):
                    q.put(list(h_ids))  # blocks when q is full

                # close pool
                for _ in range(NUM_WORKERS):
                    q.put(None)
                pool.close()
                pool.join()

                # collect best score for given combo size
                x_tick = combo_size
                x_tick2ys[x_tick].append(score_max.value)

            # compute  baselines
            baseline_acc = evaluate_linear_readout(data, net, feature_type=FEATURE_TYPE, state_is_input=True)
            random_acc = evaluate_linear_readout(data, net, feature_type=FEATURE_TYPE, state_is_random=True)

            # plot
            ys = []
            for x_tick, y in sorted(x_tick2ys.items()):
                ys.append(y)
            ys = np.array(ys).T
            x_ticks = list(sorted(x_tick2ys))
            plot_line(
                ys,
                title=f'{param_path.name}\nLinear readout at hidden state\n',
                x_axis_label='Hidden unit combination size',
                y_axis_label=f'{FEATURE_TYPE.capitalize()} Accuracy',
                x_ticks=x_ticks,
                labels=rep_names,
                y_lims=[0, 1],
                baseline_input=baseline_acc,
                baseline_random=random_acc,
            )

        break  # do not keep searching for models - regular pattern ids are defined for first model only
