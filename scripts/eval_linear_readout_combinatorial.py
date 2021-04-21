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
from collections import defaultdict

from polyominoworld.utils import get_leftout_positions, evaluate_detector_combo
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests

from ludwig.results import gen_param_paths

MIN_COMBO_SIZE = 1
NUM_WORKERS = 6
FEATURE_TYPE = 'shape'

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
            net.eval()
            h_x = net.h_x.weight.detach().numpy()  # [num hidden, num world cells]

            # set up parallel processes that read from queue and save results in shared memory + shared memory
            q = mp.Queue(maxsize=NUM_WORKERS)
            score_max = mp.Value('d')
            score_max.value = +0.0
            pool = mp.Pool(NUM_WORKERS,
                           initializer=evaluate_detector_combo,
                           initargs=(q,
                                     data,
                                     net,
                                     FEATURE_TYPE,
                                     score_max,
                                     ))

            # search all combinations of input weight patterns
            for combo_size in range(MIN_COMBO_SIZE, params.hidden_size + 1):
                print(f'Searching combo size={combo_size}')
                for h_ids in combinations(range(params.hidden_size), combo_size):
                    q.put(h_ids)  # blocks when q is full

            # close pool
            print('Closing pool')
            for _ in range(NUM_WORKERS):
                q.put(None)
            pool.close()
            print('Joining pool')
            pool.join()

            # report
            print(f'best score={score_max.value:.4f} FEATURE_TYPE={FEATURE_TYPE}')
            print()

        break  # do not keep searching for models - regular pattern ids are defined for first model only
