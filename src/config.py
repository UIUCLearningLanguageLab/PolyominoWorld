class PrintOptions:
    print_items = True
    print_confusion_matrices = True
    def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
    def print_green(skk): print("\033[92m {}\033[00m" .format(skk)) 


class Display:
    cell_size = 20


class Shape:
    master_shape_list = ['monomino', 'domino',
                  'tromino1', 'tromino2',
                  'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    master_action_list = ['appear', 'rest', 'move', 'rotate', 'flip']
    master_size_list = [1, 2, 3, 4]
    master_color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

    action_probs_list = [.00, .25, .25, .25, .25]

    color_value_dict = {'black': (-1., -1., -1.),
                        'white': (1., 1., 1.),
                        'red': (1., -1., -1.),
                        'blue': (-1., -1., -1.),
                        'green': (-1., 1., 1.),
                        'yellow': (1., 1., -1.),
                        'cyan': (-1., 1., 1.),
                        'magenta': (1., -1., 1.),
                        'grey': (0., 0., 0.),
                        'silver': (-0.5, -0.5, -0.5),
                        'teal': (0, 0.5, 0.5)
                        }

    num_shapes = len(master_shape_list)
    num_sizes = len(master_size_list)
    num_colors = len(master_color_list)
    num_actions = len(master_action_list)