"""
Experimental script to quantify how invariant the hidden states are to
rotation and location variants of a shape.

1) Combinations of input weights obtained (each called a detector),
and then made discrete by rounding to the nearest mode.
2) The hidden state is computed using these detectors.
3) a score is computed that is proportional to the number of states not shared between shapes (higher is better)

Notes:
    - each primary color channel is evaluated separately. all combinations of input weight patterns are searched,
    to identify the combination of patterns that results in highest measure of invariance.
    - measure of invariance is a product of two proportions, consequently max=1 and min=0.
    - a perfect score is not expected because networks are trained to distinguish color,
    which means that hidden state cannot be expected to abstract over (become invariant to) color.

"""

import torch
import yaml
from itertools import combinations
import multiprocessing as mp
from pathlib import Path

from polyominoworld import configs
from polyominoworld.utils import get_leftout_positions, evaluate_detector_combo
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests

from ludwig.results import gen_param_paths

MAX_COMBO_SIZE = 2
SCALE = 1.0  # scale weights so that rounding to nearest integer effectively rounds to nearest mode
NUM_WORKERS = 6
# manually specify ids of input weights that appear regular. do this for first model replication, which is loaded first
HIDDEN_IDS = [i for i in range(16)]

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
                               initializer=evaluate_detector_combo,
                               initargs=(q,
                                         data,
                                         h_x,
                                         SCALE,
                                         rgb_id,
                                         largest_avg_res,
                                         ))

                # search all combinations of input weight patterns
                for combo_size in range(1, MAX_COMBO_SIZE + 1):
                    for h_ids in combinations(HIDDEN_IDS, combo_size):
                        q.put(h_ids)  # blocks when q is full

                # close pool
                for _ in range(NUM_WORKERS):
                    q.put(None)
                pool.close()
                pool.join()

                print(f'best score={largest_avg_res.value:.4f} overall for color={color} ')
                print()

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
