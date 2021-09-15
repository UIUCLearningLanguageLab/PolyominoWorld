"""
This file defines the hyper-parameters used for each Ludwig job.
By collecting multiple values per hyper-parameter in a list in param2requests,
Ludwig will run jobs corresponding to all combination of hyper-parameter values.
Any hyper-parameter not overwritten by param2requests will be assigned its default value using param2default.

Note: Ludwig relies on the three dictionaries below to be named as-is. Do not rename them.

Organisation as of January 2021:

experiment 2a: all colors except 1
experiment 2b: all shapes except 1
experiment 2c: upper vs. lower
experiment 2d: train on half of variants, test on other halves
experiment 3: continue training exp2 models on full data, tracking speed of learning of novel examples


Organisation as of August 2021:

Experiment 1: the basic findings
Experiment 2: training 5 models for N epochs on top or bottom only (while testing the untrained half),
then another N epochs on top+bottom (while continuing to test the originally untrained half)
Experiment 3: train 5*8 models for N epochs, each set of 5 omitting one color (while testing on the untrained color),
and then continuing to train all models for N epochs on all colors
 (continuing to test only the originally omitted color)
Experiment 4: train 5*9 models for N epoch, each set of 5 omitting 1 shape (while testing on the untrained shape),
and then continuing to train all models for N epochs on all shapes
 (continuing to test only the originally omitted shape)
Experiment 5: train 5*2 models for N epochs, each set of 5 omitting half the shape variants
 (while testing on the omitted half of the variants), then continuing to train all models for N epochs on all variants
  (continuing to test only on the originally omitted variants)


hyper-parameter tuning notes (when background color='black'):

    no hyper parameters were found that allowed a 16 hidden unit model to achieve perfect accuracy,
    but adding a second layer of 12 (but not lower) hidden units results in perfect accuracy.

    best hyper-parameters for hidden size=18:
        - for batch size 4096: lr=0.01, 6.0, 0.0 num_steps=300K                           -> 1.0 accuracy at step=200K
     best hyper-parameters for hidden size=30:
        - for batch size    1: lr=0.01, 1.6, 1.0 num_steps= 1.6M                          -> 1.0 accuracy at step=1.6M
     best hyper-parameters for hidden size=32:
        - for batch size    1: lr=0.01, 2.8, 0.0 num_steps=  1M                           -> 1.0 accuracy at step=1M
        - for batch size  128: lr=0.02, 4.0, 0.0 num_steps=300K                           -> 1.0 accuracy at step=200K
        - for batch size 4096: lr=0.02, 4.4, 0.0 num_steps= 60K  nesterov momentum=0.95   -> 1.0 accuracy at step=15K
        - for batch size 8192: lr=0.01, 8.0, 0.0 num_steps=100k                           -> 1.0 accuracy at step=40K

    WARNING: num_steps interacts with cyclical learning rate schedule

    nesterov should be set high when using large batches, but 0 when batch_size=1

    it is almost impossible to get consistently perfect accuracy below 32 hidden units without batching
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

if not runs_path.exists():
    raise FileNotFoundError(f'Did not find {runs_path}. Check that your environment variable LUDWIG_MNT is correct')

# default hyper parameters with batch-size=1
param2default = {
    # model
    'load_from_checkpoint': 'none',
    'hidden_sizes': (32, ),
    'learning_rates': (0.4, 0.4, 0.4),  # start, mid, and end lr
    'batch_size': 1,  # large batch size allows convergence of much smaller models
    'num_steps': 2_000_000,  # this is matched precisely to learning rate schedule
    'weight_init': 0.01,
    'optimizer': 'SGD',
    'momenta': (0.00, 0.0, 0.0),
    'nesterov': False,  # requires momentum to be non-zero
    'weight_decay': 0.0,
    'x_type': 'world',
    'y_type': 'features',
    'criterion': 'bce',
    'hidden_activation_function': 'tanh',

    # data
    'seed': 1,
    'shuffle_sequences': True,
    'shuffle_events': False,
    'add_grayscale': False,  # adding grayscale does not help or hurt
    'bg_color': 'grey',
    'fg_colors': (
        'white',
        'black',
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

    # specific to train data
    'train_leftout_variants': '',  # is a string, and can be either "", "half1", or "half2
    'train_leftout_half': '',  # is a string, and can be either "", "upper", or "lower"
    'train_leftout_colors': (),  # empty means nothing is leftout
    'train_leftout_shapes': (),

    # specific to test data
    'test_leftout_variants': '',  # is a string, and can be either "", "half1", or "half2
    'test_leftout_half': '',  # is a string, and can be either "", "upper", or "lower"
    'test_leftout_colors': (),  # empty means nothing is leftout
    'test_leftout_shapes': (),

}

# minimal hyper-parameters used for speedy debugging/testing
param2debug = {
    'num_steps': 10_000,
}


def find_param_name(**kwargs,
                    ) -> str:
    """return the param_name that corresponds to a particular experiment-2 configuration"""

    for param_path in runs_path.glob('param_*'):
        # load param2val
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        # exclude param_name if any parameter value does not match kwargs
        if any([True if param2val[k] != v else False for k, v in kwargs.items()]):
            continue
        # exclude if any non-kwarg parameter value does not match defaults
        for k, v in param2default.items():
            if k in kwargs:
                continue
            if param2val[k] != v:
                break
        else:
            print(f'Found requested experiment-2 param_name: {param_path.name}')
            return param_path.name

    raise FileNotFoundError(f'Did not find param_name with configuration={kwargs}')


# ############################################# user enters requested parameter configuration here


# WARNING:
# a tuple with a single string must be followed by a comma for yaml to correctly identify it as a tuple, not string

param2requests = {

    'load_from_checkpoint': ['none',
                             find_param_name(train_leftout_shapes= ('monomino', )),
                             # 'param_023',
                             ],
    'test_leftout_shapes': [('domino', 'tromino1', 'tromino2', 'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5')],

}

# #############################################

if 'train_leftout_colors' in param2requests:
    if not isinstance(param2requests['train_leftout_colors'], tuple):
        raise TypeError('"train_leftout_colors" must be of type tuple')

if 'train_leftout_shapes' in param2requests:
    if not isinstance(param2requests['train_leftout_shapes'], tuple):
        raise TypeError('"train_leftout_shapes" must be of type tuple')

# check
if 'train_leftout_colors' in param2requests:
    for leftout_colors in param2requests['train_leftout_colors']:
        for lc in leftout_colors:
            if lc == param2default['bg_color']:
                raise ValueError(f'Cannot leave out bg_color. Remove "{lc}" from train_leftout_colors.')

if 'test_leftout_colors' in param2requests:
    for leftout_colors in param2requests['test_leftout_colors']:
        for lc in leftout_colors:
            if lc == param2default['bg_color']:
                raise ValueError(f'Cannot leave out bg_color. Remove "{lc}" from test_leftout_colors.')
# check
if 'nesterov' in param2requests:
    if 'optimizer' in param2requests:
        for optim in param2requests['optimizer']:
            if optim != 'SGD':
                raise ValueError(f'Cannot use nesterov momentum without SGD.')
    if 'momenta' not in param2requests:
        raise ValueError('When nesterov=True, momenta must be non-zero')
    else:
        for ma in param2requests['momenta']:
            for mom in ma:
                if mom == 0.0:
                    raise ValueError('When nesterov=True, momentum must be non-zero')


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """
    load_from_checkpoint: str
    hidden_sizes: Tuple[int]
    learning_rates: Tuple[float, float, float]
    batch_size: int
    num_steps: int
    weight_init: float
    optimizer: str
    momenta: Tuple[float, float, float]
    nesterov: bool
    weight_decay: float
    x_type: str
    y_type: str
    criterion: str
    hidden_activation_function: str

    seed: int
    shuffle_sequences: bool
    shuffle_events: bool
    add_grayscale: bool
    bg_color: str
    fg_colors: Tuple[str,]
    actions_and_probabilities: Dict[str, float]
    shapes_and_variants: Tuple[Tuple[str, Tuple[int, ]]]
    num_events_per_sequence: int

    train_leftout_variants: str
    train_leftout_half: str
    train_leftout_colors: Tuple[str]
    train_leftout_shapes: Tuple[str]

    test_leftout_variants: str
    test_leftout_half: str
    test_leftout_colors: Tuple[str]
    test_leftout_shapes: Tuple[str]

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)
