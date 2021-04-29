"""
This file defines the hyper-parameters used for each Ludwig job.
By collecting multiple values per hyper-parameter in a list in param2requests,
Ludwig will run jobs corresponding to all combination of hyper-parameter values.
Any hyper-parameter not overwritten by param2requests will be assigned its default value using param2default.

Note: Ludwig relies on the three dictionaries below to be named as-is. Do not rename them.

experiment 2a: all colors except 1
experiment 2b: all shapes except 1
experiment 2c: upper vs. lower
experiment 2d: train on half of variants, test on other halves

experiment 3: continue training exp2 models on full data, tracking speed of learning of novel examples


hyper-parameter tuning notes:
    the smallest number of hidden units that can achieve perfect accuracy within 1 million steps
    is a model with 21 hidden units. a model with 16 hidden units achieves a best accuracy  near 95%.
    however, it can achieve 100% when only red, green, and blue colors are allowed

     best hyper-parameters for hidden size=32:
        - for batch size   1: lr=2.8, weight init=0.001 -> perfect accuracy at step=600K in 3 minutes
        - for batch size 128: lr=4.0, weight init=0.01  -> perfect accuracy at step=200K in 1 minute

"""
from typing import Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import yaml

try:
    mnt = os.getenv('LUDWIG_MNT', '/media/')
except KeyError:
    raise KeyError('Did not find an environment variable called LUDWIG_MNT. '
                   'Point it to the location where the shared drive is mounted on your system,'
                   'or use `LUDWIG_MNT=<PATH TO DRIVE>` in front of your `ludwig` command')

runs_path = Path(mnt) / 'ludwig_data' / 'PolyominoWorld' / 'runs'


def is_exp2(param_path: Path,
            ) -> bool:
    """is the parameter configuration part of experiment 2?"""

    res = False

    # load param2val
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    # check if at least 1 feature was left out -> if so, configuration is from exp 2
    for k in ['leftout_variants', 'leftout_half', 'leftout_colors', 'leftout_shapes']:
        if param2val[k] not in {'', ()}:
            res = True

    print(f'Configuration {param_path.name} is in exp2={res}')

    return res


param2requests = {

}


# default hyper parameters for fast mini-batch training
param2default_batching = {
    # model
    'load_from_checkpoint': 'none',
    'hidden_size': 32,
    'hidden_activation_function': 'tanh',
    'learning_rate': 4.0,  # max learning rate in cyclical learning rate schedule
    'num_steps': 1_000_000,
    'weight_init': 0.01,  # different from non-fast parameters (originally 0.001)
    'optimizer': 'SGD',
    'x_type': 'world',
    'y_type': 'features',
    'criterion': 'bce',
    'batch_size': 128,  # large batch + large lr size speeds convergence

    # data
    'seed': 1,
    'shuffle_sequences': True,
    'shuffle_events': False,
    'allow_negative_x': False,  # much better performance if -1s are not allowed in x
    'bg_color': 'black',
    'fg_colors': (
        'white',
        'red',
        'blue',
        'green',
        'yellow',
        'cyan',
        'magenta',
    ),
    'actions_and_probabilities': (
        ('rest', 0.0),
        ('move', 1.0),
        ('rotate', 0.0),
        ('flip',  0.0),
    ),
    'shapes_and_variants': (  # which shapes and which variant should be included
        ('monomino', (0,)),
        ('domino', (0, 1)),
        ('tromino1', (0, 1)),
        ('tromino2', (0, 1, 2, 3)),
        ('tetromino1', (0,)),
        ('tetromino2', (0, 1)),
        ('tetromino3', (0, 1, 2, 3)),
        ('tetromino4', (0, 1, 2, 3, 4, 5, 6, 7)),
        ('tetromino5', (0, 1, 2, 3))
    ),
    'num_events_per_sequence': 1,  # num of events per sequence

    'leftout_variants': '',  # is a string, and can be either "", "half1", or "half2
    'leftout_half': '',  # is a string, and can be either "", "upper", or "lower"
    'leftout_colors': (),  # empty means nothing is leftout
    'leftout_shapes': (),

}

# default hyper parameters with batch-size=1
param2default = {
    # model
    'load_from_checkpoint': 'none',
    'hidden_size': 32,
    'hidden_activation_function': 'tanh',
    'learning_rate': 2.8,  # max learning rate in cyclical learning rate schedule
    'num_steps': 1_000_000,
    'weight_init': 0.01,
    'optimizer': 'SGD',
    'x_type': 'world',
    'y_type': 'features',
    'criterion': 'bce',
    'batch_size': 1,  # large batch size only marginally speeds convergence

    # data
    'seed': 1,
    'shuffle_sequences': True,
    'shuffle_events': False,
    'allow_negative_x': False,  # much better performance if -1s are not allowed in x
    'bg_color': 'black',
    'actions_and_probabilities': (
        ('rest', 0.0),
        ('move', 1.0),
        ('rotate', 0.0),
        ('flip',  0.0),
    ),
    'shapes_and_variants': (  # which shapes and which variant should be included
        ('monomino', (0,)),
        ('domino', (0, 1)),
        ('tromino1', (0, 1)),
        ('tromino2', (0, 1, 2, 3)),
        ('tetromino1', (0,)),
        ('tetromino2', (0, 1)),
        ('tetromino3', (0, 1, 2, 3)),
        ('tetromino4', (0, 1, 2, 3, 4, 5, 6, 7)),
        ('tetromino5', (0, 1, 2, 3))
    ),
    'fg_colors': (
        'white',
        'red',
        'blue',
        'green',
        'yellow',
        'cyan',
        'magenta',
    ),
    'num_events_per_sequence': 1,  # num of events per sequence

    'leftout_variants': '',  # is a string, and can be either "", "half1", or "half2
    'leftout_half': '',  # is a string, and can be either "", "upper", or "lower"
    'leftout_colors': (),  # empty means nothing is leftout
    'leftout_shapes': (),

}

# minimal hyper-parameters used for speedy debugging/testing
param2debug = {
    'num_steps': 10_000,
}

# check
if 'leftout_colors' in param2requests:
    for leftout_colors in param2requests['leftout_colors']:
        for lc in leftout_colors:
            if lc == param2default['bg_color']:
                raise ValueError(f'Cannot leave out bg_color. Remove "{lc}" from leftout_colors.')


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """
    load_from_checkpoint: str
    hidden_size: int
    hidden_activation_function: str
    learning_rate: float
    num_steps: int
    weight_init: float
    optimizer: str
    x_type: str
    y_type: str
    criterion: str
    batch_size: int

    seed: int
    shuffle_sequences: bool
    shuffle_events: bool
    allow_negative_x: bool
    bg_color: str
    fg_colors: Tuple[str,]
    actions_and_probabilities: Dict[str, float]
    shapes_and_variants: Tuple[Tuple[str, Tuple[int, ]]]
    num_events_per_sequence: int

    leftout_variants: str
    leftout_half: str
    leftout_colors: Tuple[str]
    leftout_shapes: Tuple[str]

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)
