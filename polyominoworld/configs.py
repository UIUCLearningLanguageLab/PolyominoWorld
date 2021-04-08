"""
This module contains settings that control all aspects of the program,
that are not of direct research interest, e.g. constants, defaults, flow-control, type checking, etc.
"""
from typing import List


class Try:
    max = 100  # max num tries to find legal position


class Training:
    gpu = False  # faster on cpu


class ArgCheck:
    x_type = {'world', 'hidden'}
    y_type = {'world', 'features'}


class Display:
    height = 1_800
    width = 2_000

    world_rectangle_size = 40
    world_layer_rectangle_size = 20
    world_grid_size = 200
    hidden_layer_rectangle_size = 40
    feature_layer_rectangle_size = 40

    feature_layer_spacing = 4
    hidden_layer_spacing = 4

    font_xs = "Arial 10 bold"
    font_s = "Arial 12 bold"
    font_m = "Arial 14 bold"
    font_l = "Arial 16 bold"
    font_xl = "Arial 16 bold"

    # x and y position on the display
    condition2position = {
        'World State': (100, 40, "World State"),
        'Predicted World State': (1000, 120, "Predicted World State"),

        'World Layer Activations': (600, 40, "Input Layer"),
        'Predicted World Layer Activations': (800, 10, "Output Layer"),
        'World Layer Weights': (280, 40, ""),
        'Predicted World Layer Weights': (800, 40, ""),

        'Hidden Layer Activations': (1000, 20, "Hidden Layer"),
        'Hidden Layer Weights': (1000, 20, ""),

        'Predicted Feature Activations': (1600, 20, "Output Layer"),
        'Predicted Feature Weights': (1600, 20, ""),
    }

    color_bg = "white"
    color_bg_button = 'white'
    color_text_fill = "white"


class Evaluation:
    epoch_interval = 10
    means_only = False  # True if collecting summaries only (e.g. means)


class World:
    """world hyper-parameters that are not intended to be changed"""

    # warning: do not change world size when intending to evaluate previous models trained with original size
    # warning: max_x and max_y must be divisible by two (because world is divided into two halves)
    max_x = 8
    max_y = 8
    bounds = [0, max_x,  # x min, x max
              0, max_y,  # y min, y max
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

    master_shapes: List[str] = [
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

    master_sizes: List[str] = [1, 2, 3, 4]  # size = number of active cells for a shape
    master_colors: List[str] = [c for c in color2rgb]
    master_actions: List[str] = [a for a in action2directions]
    feature_type2values = {
        'shape': master_shapes,
        'size': master_sizes,
        'color': master_colors,
        'action': master_actions,
    }


class Figs:
    lw = 1
    ax_label_fs = 14
    leg_fs = 8
    dpi = 163
