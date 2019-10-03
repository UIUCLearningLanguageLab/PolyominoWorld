from src import config


class World:

    def __init__(self):
        self.history_list = []
        self.event_counter = None
        self.num_scenes = config.World.num_scenes
        self.scene_length = config.World.scene_length
        self.num_rows = config.World.num_rows
        self.num_columns = config.World.num_columns
        self.num_shapes = config.World.num_shapes

        self.shape_id_counter = None
        self.occupied_cell_dict = None
        self.current_shape_list = None
        self.shape_dict = None
        self.turn_counter = None

        self.num_features = 0
        self.feature_list = []
        self.feature_index_dict = {}
        self.color_list = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
        self.color_list = ['black']
        self.shapes_list = ['Monomino', 'Domino',
                            'Tromino1', 'Tromino2',
                            'Tetromino1', 'Tetromino2', 'Tetromino3', 'Tetromino4', 'Tetromino5']
        self.color_value_dict = {'black':   (-1., -1., -1.),
                                 'white':   (1., 1., 1.),
                                 'red':     (1., -1., -1.),
                                 'blue':    (-1., 1., -1.),
                                 'green':   (-1., -1., 1.),
                                 'yellow':  (1., -1., 1.),
                                 'cyan':    (-1., 1., 1.),
                                 'magenta': (1., 1., -1.),
                                 'grey': (0., 0., 0.),
                                 'silver': (-.5, -.5, -.5)
                                 }
        self.sizes_list = [1, 2, 3, 4]
        self.generate_label_indexes()

    def init_world(self, event_counter):
        self.event_counter = event_counter
        self.shape_id_counter = 0
        self.occupied_cell_dict = {}
        self.current_shape_list = []
        self.shape_dict = {}
        self.turn_counter = 0

    def add_shape_to_world(self, shape, shape_id_counter):
        self.current_shape_list.append(shape)
        self.shape_dict[shape_id_counter] = shape

    def next_turn(self):
        for i in range(self.num_shapes):
            self.current_shape_list[i].take_turn()
            self.turn_counter += 1

    def save_world_state(self, file_name):

        outfile = open(file_name, 'a')
        output_string = "{},{},".format(self.event_counter, self.turn_counter)

        for i in range(self.num_shapes):
            output_string += "{},{},{},{},{},{},{},".format(self.current_shape_list[i].name,
                                                            self.current_shape_list[i].size,
                                                            self.current_shape_list[i].color,
                                                            self.current_shape_list[i].current_variant,
                                                            self.current_shape_list[i].position[0],
                                                            self.current_shape_list[i].position[1],
                                                            self.current_shape_list[i].action_choice)

        r_string = ""
        g_string = ""
        b_string = ""

        for i in range(self.num_rows + 2):
            for j in range(self.num_columns + 2):

                if i == 0 or i == self.num_rows+1 or j == 0 or j == self.num_columns+1:
                    color = 'grey'
                else:
                    if (i, j) in self.occupied_cell_dict:
                        shape_id = self.occupied_cell_dict[(i, j)]
                        color = self.shape_dict[shape_id].color
                    else:
                        color = 'silver'

                values = self.color_value_dict[color]
                r_string += "{:s},".format(str(values[0]))
                g_string += "{:s},".format(str(values[1]))
                b_string += "{:s},".format(str(values[2]))
                j += 1
            i += 1
        output_string += r_string[:-1] + "," + g_string[:-1] + "," + b_string[:-1] + '\n'
        outfile.write(output_string)

    def reset(self):
        self.init_world(self.event_counter)

    def generate_label_indexes(self):
        for shape in self.shapes_list:
            self.feature_list.append(shape)
            self.feature_index_dict[shape] = self.num_features
            self.num_features += 1
        for size in self.sizes_list:
            self.feature_list.append(size)
            self.feature_index_dict[size] = self.num_features
            self.num_features += 1
        for color in self.color_list:
            self.feature_list.append(color)
            self.feature_index_dict[color] = self.num_features
            self.num_features += 1
