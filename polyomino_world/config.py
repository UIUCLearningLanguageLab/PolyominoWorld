
class PrintOptions:
    print_items = True
    print_confusion_matrices = True
    def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
    def print_green(skk): print("\033[92m {}\033[00m" .format(skk))


class Display:
    cell_size = 20


class Shape:
    master_feature_type_list = ['shape', 'size', 'color', 'action']
    master_feature_type_index_dict = {'shape': 0, 'size': 1, 'color': 2, 'action': 3}
    num_master_feature_types = 4

    master_shape_list = ['monomino', 'domino', 'tromino1', 'tromino2',
                         'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    master_size_list = [1, 2, 3, 4]

    master_color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    color_value_dict = {'black': (-1., -1., -1.),
                        'white': (1., 1., 1.),

                        'red': (1., -1., -1.),
                        'blue': (-1., -1., 1.),
                        'green': (-1., 1., -1.),

                        'yellow': (1., 1., -1.),
                        'cyan': (-1., 1., 1.),
                        'magenta': (1., -1., 1.),

                        'grey': (0., 0., 0.),
                        }

    master_action_list = ['rest', 'move', 'rotate', 'flip']
    action_prob_list = [.0, 1.0, .0, .0]  # [.25, .25, .25, .25]

    master_feature_type_size_dict = {'shape': len(master_shape_list),
                                     'size': len(master_size_list),
                                     'color': len(master_color_list),
                                     'action': len(master_action_list)}
