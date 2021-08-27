import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml

from polyominoworld import configs
from polyominoworld.params import Params


def get_leftout_positions(leftout_half: str,
                          ) -> List[Tuple[int, int]]:
    """get positions in world that are in leftout half of the world"""

    all_positions = [(x, y) for x, y in
                     itertools.product(range(configs.World.max_x), range(configs.World.max_y))]

    if leftout_half == 'lower':
        return [(x, y) for x, y in all_positions
                if y < configs.World.max_y / 2]
    elif leftout_half == 'upper':
        return [(x, y) for x, y in all_positions
                if y >= configs.World.max_y / 2]
    elif leftout_half == '':
        return []  # nothing is leftout
    else:
        raise AttributeError('Invalid arg to leftout_half')


def calc_world_vector_size(add_grayscale: bool,
                           ) -> int:
    num_cells = configs.World.max_x * configs.World.max_x
    if add_grayscale:
        num_channels = 4
    else:
        num_channels = 3  # rgb
    return num_cells * num_channels


def get_test_data_kwargs(param2val: Dict[str, Any],
                         ) -> Dict[str, Any]:
    """
    return keyword arguments for test data creation function that specify what is leftout from test data.

    Note:
        There are 3 options for specifying what is leftout (ordered by priority):
            1) custom instructions for what should be leftout in test data, or
            2) based on what is leftout from training data, or
            3) based on what is leftout from pretraining dataset, or
            otherwise, nothing is leftout
    """
    # init with defaults specifying nothing is leftout
    res = {'leftout_colors': (),
           'leftout_shapes': (),
           'leftout_variants': '',
           'leftout_positions': get_leftout_positions(''),
           }

    params: Params = Params.from_param2val(param2val)

    param_names_test_leftout = ['test_leftout_colors',
                                'test_leftout_shapes',
                                'test_leftout_variants',
                                'test_leftout_half',
                                ]
    param_names_train_leftout = ['train_leftout_colors',
                                 'train_leftout_shapes',
                                 'train_leftout_variants',
                                 'train_leftout_half',
                                 ]

    # option 1: use manual instructions for leaving out data from test split
    if any([True if param2val[p] else False for p in param_names_test_leftout]):
        res['leftout_colors'] = params.test_leftout_colors
        res['leftout_shapes'] = params.test_leftout_shapes
        res['leftout_variants'] = params.test_leftout_variants
        res['leftout_positions'] = get_leftout_positions(params.test_leftout_half)

    # option 2: leave out data not included in train data
    elif any([True if param2val[p] else False for p in param_names_train_leftout]):
        if params.train_leftout_colors:
            res['leftout_colors'] = tuple([c for c in configs.World.master_colors
                                           if c not in params.train_leftout_colors])
        if params.train_leftout_shapes:
            res['leftout_shapes'] = tuple([c for c in configs.World.master_shapes
                                           if c not in params.train_leftout_shapes])
        if params.train_leftout_variants:
            res['leftout_variants'] = {'half1': 'half2', 'half2': 'half1'}[params.train_leftout_variants]
        if params.train_leftout_half:
            res['leftout_positions'] = get_leftout_positions(
                {'upper': 'lower', 'lower': 'upper'}[params.train_leftout_half])

    # option 3: leave out data not included in pre-training data
    elif param2val['load_from_checkpoint']:
        param_path_pretraining = Path(param2val['project_path']) / 'runs' / param2val['load_from_checkpoint']
        with (param_path_pretraining / 'param2val.yaml').open('r') as f:
            param2val_pretraining = yaml.load(f, Loader=yaml.FullLoader)
        params_pretrain: Params = Params.from_param2val(param2val_pretraining)
        print('Leftout during pre-training:')
        print([f'{k}={v}' for k, v in param2val_pretraining.items() if k.startswith('leftout')])
        res['leftout_colors'] = tuple([c for c in configs.World.master_colors if c not in params_pretrain.train_leftout_colors])
        res['leftout_shapes'] = tuple([c for c in configs.World.master_shapes if c not in params_pretrain.train_leftout_shapes])
        res['leftout_variants'] = {'half1': 'half2', 'half2': 'half1'}[params_pretrain.train_leftout_variants]
        res['leftout_positions'] = get_leftout_positions({'upper': 'lower', 'lower': 'upper'}[params_pretrain.train_leftout_half])

    # option 4: do not leave anything out in test data
    else:
        return res

    return res
