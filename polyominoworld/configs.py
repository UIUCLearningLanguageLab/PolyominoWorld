"""
This module contains settings that control all aspects of the program,
that are not of direct research interest, e.g. constants, defaults, flow-control, type checking, etc.
"""
from typing import List
from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    embedding_projector = root / 'embedding_projector'  # tsv files for tensorflow embedding projector app


class Try:
    max = 100  # max num tries to find legal position


class Device:
    gpu = True  # faster on cpu below batch size=4096 and hidden size=16


class ArgCheck:
    x_type = {'world', 'hidden'}
    y_type = {'world', 'features'}


class Display:
    height = 2_000
    width = 2_000

    world_rectangle_size = 40
    left_section_rectangle_size = 20
    world_grid_size = 200
    middle_section_rectangle_size = 40
    right_section_rectangle_size = 40

    right_section_spacing = 4
    middle_section_spacing = 4

    font_xs = "Arial 10 bold"
    font_s = "Arial 12 bold"
    font_m = "Arial 14 bold"
    font_l = "Arial 16 bold"
    font_xl = "Arial 16 bold"

    # x and y position on the display
    top_offset = 40
    condition2position = {
        'World State': (100, top_offset, "World State"),
        'Predicted World State': (1000, top_offset, "Predicted World State"),

        'World Layer Activations': (600, top_offset, "Input Layer"),
        'Predicted World Layer Activations': (800, top_offset, "Output Layer"),
        'World Layer Weights': (280, top_offset, ""),
        'Predicted World Layer Weights': (800, top_offset, ""),

        'Hidden Layer Activations': (1000, top_offset, "Hidden Layer"),
        'Hidden Layer Weights': (1000, top_offset, ""),

        'Predicted Feature Activations': (1600, top_offset, "Output Layer"),
        'Predicted Feature Weights': (1600, top_offset, ""),
    }

    color_bg = "white"
    color_bg_button = 'white'
    color_text_fill = "white"


class Evaluation:
    step_interval = 10_000
    skip_test_data = False


class World:
    """world hyper-parameters that are not intended to be changed"""

    # warning: do not change world size when intending to evaluate previous models trained with original size
    # warning: max_x and max_y must be divisible by two (because world is divided into two halves)
    max_x = 8
    max_y = 8
    bounds = [0, max_x,  # x min, x max
              0, max_y,  # y min, y max
              ]

    color2rgb = {'black':   [-1.0, -1.0, -1.0],
                 'white':   [+1.0, +1.0, +1.0],
                 'red':     [+1.0, -1.0, -1.0],
                 'blue':    [-1.0, -1.0, +1.0],
                 'green':   [-1.0, +1.0, -1.0],
                 'yellow':  [+1.0, +1.0, -1.0],
                 'cyan':    [-1.0, +1.0, +1.0],
                 'magenta': [+1.0, -1.0, +1.0],
                 'grey':    [+0.0, +0.0, +0.0],
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
    ax_font_size = 14
    leg_font_size = 8
    dpi = 163
    title_font_size = 8
    tick_font_size = 8

    NO_PRE_TRAINING_STRING = 'no pre-training'
