"""
Experimental script to quantify how invariant the hidden states are to
rotation and location variants of a shape.

1) Combinations of input weights are obtained (each called a detector),
and then made discrete by rounding to the nearest mode.
2) The hidden state is computed using these detectors.
3) a score is computed that is proportional to the number of states not shared between shapes (higher is better).
the score cna be interpreted as a measure of "separability" of shapes at the hidden layer

Notes:
    - each primary color channel is evaluated separately when RGB_ID is not None.
     all combinations of input weight patterns are searched,
    to identify the combination of patterns that results in highest measure of invariance.
    - scores range form 0 to 1, with higher better

Findings:
    - a perfect score can be achieved when using all detectors.
     this means that shapes are perfectly separable at hidden layer, but only within a color channel.
     consequently, even in the case that the score for each color channel is perfect,
     a model may not be perfectly able to classify shapes, because the separability of shapes must hold also
     across color-channels, not just within.
     it is likely a model will first learn unique strategies for separating shapes within each color channel,
     resulting in perfect score,
    but never be able to combine the 3 solutions into a single solution that works equally well across channels.
    - when using only 2 detectors, checkerboard detectors are best,
    but when using more than 2 detectors, non-regular non-checkerboard detectors give highest score
    - there is less overlap of shape states at hidden layer when analyzing across color channels,
    because including each color channel means including -1 in color channels in which the primary color is not,
    and these additional -1 allow more unique ways of representing each instance

"""

import torch
import yaml
from itertools import combinations
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import numpy as np

from polyominoworld.utils import get_leftout_positions, evaluate_detector_combo
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.figs import plot_line

from ludwig.results import gen_param_paths

MIN_COMBO_SIZE = 16
SCALE = 1.0  # scale weights so that rounding to nearest integer effectively rounds to nearest mode
NUM_WORKERS = 6
RGB_ID = None  # None to evaluate all three color channels
ACROSS_TIME = True
ACROSS_SHAPE = False

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

        if RGB_ID is not None:
            if {0: 'red', 1: 'green', 2: 'blue'}[RGB_ID] not in params.colors:
                raise ValueError(f'The requested color channel is not in params.colors')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        paths_to_net = list(sorted(param_path.rglob(f'model{"_*" if ACROSS_TIME else ""}.pt')))
        if not paths_to_net:
            raise FileNotFoundError('Did not find any model files')

        assert MIN_COMBO_SIZE <= params.hidden_size

        x_tick2ys = defaultdict(list)
        for path_to_net in paths_to_net:

            print(f'Loading net from {path_to_net}')

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()
            h_x = net.h_x.weight.detach().numpy()  # [num hidden, num world cells]

            # set up parallel processes that read from queue and save results in shared memory + shared memory
            q = mp.Queue(maxsize=NUM_WORKERS)
            largest_avg_res = mp.Value('d')
            largest_avg_res.value = +0.0
            pool = mp.Pool(NUM_WORKERS,
                           initializer=evaluate_detector_combo,
                           initargs=(q,
                                     data,
                                     h_x,
                                     SCALE,
                                     largest_avg_res,
                                     RGB_ID,
                                     ACROSS_SHAPE,
                                     ))

            # search all combinations of input weight patterns
            jump_to_next_model = False
            for combo_size in range(MIN_COMBO_SIZE, params.hidden_size + 1):
                print(f'Searching combo size={combo_size}')
                for h_ids in combinations(range(params.hidden_size), combo_size):
                    q.put(h_ids)  # blocks when q is full

                    if largest_avg_res.value > 0.99:
                        jump_to_next_model = True
                        break
                if jump_to_next_model:
                    break

            # close pool
            for _ in range(NUM_WORKERS):
                q.put(None)
            pool.close()
            pool.join()

            if largest_avg_res.value == 0.0:
                print('Skipping')
                continue

            # report
            if RGB_ID is not None:
                condition = 'within'
                tmp = f'={RGB_ID}'
            else:
                condition = 'across'
                tmp = 's'
            print(f'best score={largest_avg_res.value:.4f} {condition} color channel{tmp}')
            print()

            x_tick = int(path_to_net.stem.lstrip('model_'))
            x_tick2ys[x_tick].append(largest_avg_res.value)

        # plot
        ys = []
        for x_tick, y in sorted(x_tick2ys.items()):
            ys.append(y)
        ys = np.array(ys).T
        x_ticks = list(sorted(x_tick2ys))
        plot_line(
            ys,
            title=f'across_shape={ACROSS_SHAPE}\ncolor channel={RGB_ID}',
            x_axis_label='Evaluation Epoch',
            y_axis_label='Hidden state overlap',
            x_ticks=x_ticks,
            labels=[],
            y_lims=[0, 1]
        )


        break  # do not keep searching for models - regular pattern ids are defined for first model only
