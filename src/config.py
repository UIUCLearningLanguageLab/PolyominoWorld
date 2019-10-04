class PrintOptions:
    print_items = True
    print_confusion_matrices = True
    def print_red(skk): print("\033[91m {}\033[00m" .format(skk))
    def print_green(skk): print("\033[92m {}\033[00m" .format(skk)) 


class Display:
    cell_size = 20


class World:
    num_rows = 10
    num_columns = 10


class Shape:
    shape_list = ['monomino', 'domino',
                  'tromino1', 'tromino2',
                  'tetromino1', 'tetromino2', 'tetromino3', 'tetromino4', 'tetromino5']
    action_list = ['rest', 'move', 'rotate', 'flip']
    size_list = [1, 2, 3, 4]
    color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    action_probs_list = [.25, .25, .25, .25]
    color_value_dict = {'black': (-1., -1., -1.),
                        'white': (1., 1., 1.),
                        'red': (1., -1., -1.),
                        'blue': (-1., 1., -1.),
                        'green': (-1., -1., 1.),
                        'yellow': (1., -1., 1.),
                        'cyan': (-1., 1., 1.),
                        'magenta': (1., 1., -1.),
                        'grey': (0., 0., 0.),
                        'silver': (-0.5, -0.5, -0.5)
                        }
    num_shapes = len(shape_list)
    num_sizes = len(size_list)
    num_colors = len(color_list)
    num_actions = len(action_list)
    y_size = num_shapes + num_sizes + num_colors + num_actions


class Data:
    def __init__(self,id_number, name, size, color, image_matrix, label_matrix):
        self.id_number = id_number
        self.name = name
        self.size = size
        self.color = color
        self.image_matrix = image_matrix
        self.label_matrix = label_matrix
