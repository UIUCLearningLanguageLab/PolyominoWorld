"""
This module contains settings that control all aspects of the program,
that are not of direct research interest, e.g. constants, defaults, flow-control, type checking, etc.
"""


class Try:
    max = 100  # max num tries to find legal position


class Training:
    gpu = False


class ArgCheck:
    x_type = {'world', 'hidden'}
    y_type = {'world', 'features'}


class PrintOptions:
    print_items = True
    print_confusion_matrices = True
    def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
    def print_green(skk): print("\033[92m {}\033[00m" .format(skk))


class Display:
    cell_size = 20


class Evaluation:
    epoch_interval = 1
    verbose = False
    skip = False


class World:
    """world hyper-parameters that are not intended to be changed"""

    num_rows = 8
    num_cols = 8
    bounds = [0, num_cols,  # x min, x max because columns are typically depicted left-right
              0, num_rows,  # y min, y max because rows are typically depicted bottom-top
              ]

    color2rgb = {'black': (-1., -1., -1.),
                 'white': (1., 1., 1.),
                 'red': (1., -1., -1.),
                 'blue': (-1., -1., 1.),
                 'green': (-1., 1., -1.),
                 'yellow': (1., 1., -1.),
                 'cyan': (-1., 1., 1.),
                 'magenta': (1., -1., 1.),
                 'grey': (0., 0., 0.),
                 }

    action2directions = {
        'rest': None,
        'move': [(0, 1), (0, -1), (-1, 0), (1, 0)],
        'rotate': [0, 1],
        'flip': [0, 1],
    }

    master_shapes = [
        'monomino',
        'domino',
        'tromino1',
        'tromino2',
        'tetromino1',
        'tetromino2',
        'tetromino3',
        'tetromino4',
        'tetromino5',
    ]

    master_sizes = [1, 2, 3, 4]  # size = number of active cells for a shape
    master_colors = [c for c in color2rgb]
    master_actions = [a for a in action2directions]
    feature_type2values = {
        'shape': master_shapes,
        'size': master_sizes,
        'color': master_colors,
        'action': master_actions,
    }


