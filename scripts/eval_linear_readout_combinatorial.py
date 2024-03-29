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
from typing import Tuple
from collections import defaultdict

from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.figs import plot_lines
from polyominoworld.evaluate import evaluate_linear_readout
from polyominoworld import configs

from ludwig.results import gen_param_paths

MIN_COMBO_SIZE = 16
HIDDEN_LAYER_ID = 0
NUM_WORKERS = 6
FEATURE_TYPE = 'shape'


def init(data_,
         net_,
         feature_type_,
         score_max_,
         lock_,
         ):
    global data, net, feature_type, score_max, lock
    data = data_
    net = net_
    feature_type = feature_type_
    score_max = score_max_
    lock = lock_


def evaluate_detector_combo(h_ids_: Tuple[int],
                            ) -> None:
    """
    compute linear readout accuracy for given detector combination
    """

    from polyominoworld.evaluate import evaluate_linear_readout

    score = evaluate_linear_readout(data, net, feature_type, HIDDEN_LAYER_ID, list(h_ids_))

    # report + update shared memory
    if score > score_max.value:
        lock.acquire()
        score_max.value = score
        print(f'score={score :.4f} | h_ids={h_ids_} ')
        lock.release()

    return None


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
        data = DataSet(world.generate_sequences(leftout_colors=(),
                                                leftout_shapes=(),
                                                leftout_variants='',
                                                leftout_positions=get_leftout_positions(''),
                                                ),
                       seed=params.seed,
                       shuffle_events=params.shuffle_events,
                       shuffle_sequences=params.shuffle_sequences,
                       name='re-generated')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        paths_to_net = list(sorted(param_path.rglob(f'model.pt')))
        if not paths_to_net:
            raise FileNotFoundError('Did not find any model files')

        rep_names = [p.name for p in param_path.glob('*') if p.is_dir()]

        net = Network(params)

        hidden_size = params.hidden_sizes[HIDDEN_LAYER_ID]
        assert MIN_COMBO_SIZE <= hidden_size

        x_tick2ys = defaultdict(list)
        for path_to_net in paths_to_net:

            print(f'Loading net from {path_to_net}')

            # load weights into net
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.requires_grad_(False)
            net.eval()

            # search all combinations of hidden units
            for combo_size in range(MIN_COMBO_SIZE, hidden_size + 1):

                # set up parallel processes that read from queue and save results in shared memory + shared memory
                q = mp.Queue(maxsize=NUM_WORKERS)
                score_max = mp.Value('d')
                score_max.value = +0.0
                lock = mp.Lock()
                pool = mp.Pool(NUM_WORKERS,
                               initializer=init,
                               initargs=(data,
                                         net,
                                         FEATURE_TYPE,
                                         score_max,
                                         lock,
                                         ))

                # do the work
                print(f'Searching combo size={combo_size}')
                pool.map(evaluate_detector_combo, combinations(range(hidden_size), combo_size))

                # close pool
                for _ in range(NUM_WORKERS):
                    q.put(None)
                pool.close()
                pool.join()

                # collect best score for given combo size
                x_tick = combo_size
                x_tick2ys[x_tick].append(score_max.value)

                if score_max.value > 0.99:
                    break  # no need to keep searching

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
                x_axis_label='Hidden unit combination size',
                y_axis_label=f'{FEATURE_TYPE.capitalize()} Accuracy',
                x_ticks=x_ticks,
                labels=rep_names,
                y_lims=[0, 1],
                baseline_input=baseline_acc,
                baseline_random=random_acc,
            )

        break  # do not keep searching for models - regular pattern ids are defined for first model only
