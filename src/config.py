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
    num_scenes = 1
    scene_length = 5
    num_shapes = 1
    save_states = True


class Shape:

    shape_list = ['Monomino', 'Domino',
                  'Tromino1', 'Tromino2',
                  'Tetromino1', 'Tetromino2', 'Tetromino3', 'Tetromino4', 'Tetromino5']
    shape_sizes = [1, 2, 3, 4]
    shape_probs_dict = [10, 10, 5, 5, 2, 2, 2, 2, 2]
    shape_color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    # probability of: rest, move, rotate, flip
    action_probs_list = [.25, .25, .25, .25]
    action_probs_dict = {'Monomino':    [.25, .25, .25, .25],
                     'Domino':      [.25, .25, .25, .25],
                     'Tromino1':    [.25, .25, .25, .25],
                     'Tromino2':    [.25, .25, .25, .25],
                     'Tetromino1':  [.25, .25, .25, .25],
                     'Tetromino2':  [.25, .25, .25, .25],
                     'Tetromino3':  [.25, .25, .25, .25],
                     'Tetromino4':  [.25, .25, .25, .25],
                     'Tetromino5':  [.25, .25, .25, .25]}


class Data:
    def __init__(self,id_number, name, size, color, image_matrix, label_matrix):
        self.id_number=id_number
        self.name=name
        self.size=size
        self.color=color
        self.image_matrix=image_matrix
        self.label_matrix=label_matrix
