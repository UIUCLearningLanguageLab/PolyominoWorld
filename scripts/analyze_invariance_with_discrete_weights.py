"""
Experimental script to quantify how invariant the hidden states are to
rotation and location variants of a shape.

1) Input weights are first filtered based on whether they exhibit a regular pattern,
and then made discrete by rounding to the nearest mode.
2) The hidden state is computed using these filtered, discrete weights.
3) a measure of invariance is computed using two terms:
    a) proportion of states states shared within a shape
    b) proportion of states not shared between shapes

Notes:
    - each primary color channel is evaluated separately. all combinations of input weight patterns are searched,
    to identify the combination of patterns that results in highest measure of invariance.
    - measure of invariance is a product of two proportions, consequently max=1 and min=0.
    - a perfect score is not expected because networks are trained to distinguish color,
    which means that hidden state cannot be expected to abstract over (become invariant to) color.

"""

import torch
import yaml
import numpy as np
from itertools import combinations
import multiprocessing as mp

from polyominoworld import configs
from polyominoworld.utils import get_leftout_positions, calc_terms1_and_terms2
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default

from ludwig.results import gen_param_paths

SCALE = 1.1  # scale weights so that rounding to nearest integer effectively rounds to nearest mode
NUM_WORKERS = 4
# manually specify ids of input weights that appear regular. do this for first model replication, which is loaded first
REGULAR_PATTERN_IDS = [3, 4, 5, 6, 7, 11, 12, 14, 16, 20, 22, 23, 24, 25, 29, 30, 31][::-1]
# REGULAR_PATTERN_IDS = [25, 30, 31]
# REGULAR_PATTERN_IDS = [6, 11]
PLOT_WEIGHTS = False
PLOT_STATES = False

if __name__ == '__main__':

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # only use one color channel
        for rgb_id, color in enumerate(['red', 'green', 'blue']):

            # use all locations, rotations, and shapes, but only one primary color
            world = World(params)
            data = DataSet(world.generate_sequences(leftout_colors=tuple([c for c in configs.World.master_colors
                                                                          if c != color]),
                                                    leftout_shapes=('', ),
                                                    leftout_variants='',
                                                    leftout_positions=get_leftout_positions(''),
                                                    ),
                           params,
                           name='re-generated')

            # multiple models may exist for the same hyper-parameter configuration - iterate over each
            for path_to_net in list(param_path.rglob('model.pt'))[:1]:

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
                               initializer=calc_terms1_and_terms2,
                               initargs=(q, data, h_x, SCALE, rgb_id, largest_avg_res, PLOT_WEIGHTS, PLOT_STATES))

                # search all combinations of input weight patterns
                for combo_size in range(1, len(REGULAR_PATTERN_IDS) + 1):
                    for h_ids in combinations(REGULAR_PATTERN_IDS, combo_size):
                        q.put(h_ids)  # blocks when q is full

                # close pool
                for _ in range(NUM_WORKERS):
                    q.put(None)
                pool.close()
                pool.join()

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
