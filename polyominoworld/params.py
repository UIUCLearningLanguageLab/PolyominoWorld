"""
This file defines the hyper-parameters used for each Ludwig job.
By collecting multiple values per hyper-parameter in a list in param2requests,
Ludwig will run jobs corresponding to all combination of hyper-parameter values.
Any hyper-parameter not overwritten by param2requests will be assigned its default value using param2default.

Note: Ludwig relies on the three dictionaries below to be named as-is. Do not rename them.

"""
from typing import Dict, Tuple
from dataclasses import dataclass


param2requests = {
    'hidden_size': [8, 16],
    'learning_rate': [0.3],
    'num_epochs': [10],
}


# default hyper parameters
param2default = {
    # model
    'hidden_size': 16,
    'hidden_activation_function': 'tanh',
    'learning_rate': 0.3,
    'num_epochs': 10,
    'weight_init': 0.00001,
    'optimizer': 'SGD',
    'x_type': 'world',
    'y_type': 'features',
    'criterion': 'bce',

    # data
    'seed': 1,
    'shuffle_sequences': True,
    'shuffle_events': False,
    'bg_color': 'grey',
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
    'colors': (
        'red',
        'green',
        'blue',
    ),

    # TODO add position constraint (e.g. bounds)

    'num_events_per_sequence': 1,  # num of events per sequence

}

# minimal hyper-parameters used for speedy debugging/testing
param2debug = {
    'num_epochs': 3,
}


@dataclass
class Params:
    """
    this object is loaded at the start of job.main() by calling Params.from_param2val(),
    and is populated by Ludwig with hyper-parameters corresponding to a single job.
    """
    hidden_size: int
    hidden_activation_function: str
    learning_rate: float
    num_epochs: int
    weight_init: float
    optimizer: str
    x_type: str
    y_type: str
    criterion: str

    seed: int
    shuffle_sequences: bool
    shuffle_events: bool
    bg_color: str
    actions_and_probabilities: Dict[str, float]
    shapes_and_variants: Tuple[Tuple[str, Tuple[int, ]]]
    colors: Tuple[str]
    num_events_per_sequence: int

    @classmethod
    def from_param2val(cls, param2val):
        kwargs = {k: v for k, v in param2val.items()
                  if k not in ['job_name', 'param_name', 'save_path', 'project_path']}
        return cls(**kwargs)