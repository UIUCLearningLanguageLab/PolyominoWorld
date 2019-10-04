from src import config
from src import shapes


class World:

    def __init__(self):
        self.history_list = []
        self.event_counter = None
        self.num_rows = config.World.num_rows
        self.num_columns = config.World.num_columns
        self.num_shapes = 1

        self.shape_id_counter = None
        self.occupied_cell_dict = None
        self.current_shape_list = None
        self.shape_dict = None
        self.turn_counter = None

        self.num_features = 0
        self.feature_list = []
        self.feature_index_dict = {}

        self.master_shape_dict = {'monomino': shapes.Monomino(self),
                                  'domino': shapes.Domino(self),
                                  'tromino1': shapes.Tromino1(self),
                                  'tromino2': shapes.Tromino2(self),
                                  'tetromino1': shapes.Tetromino1(self),
                                  'tetromino2': shapes.Tetromino2(self),
                                  'tetromino3': shapes.Tetromino3(self),
                                  'tetromino4': shapes.Tetromino4(self),
                                  'tetromino5': shapes.Tetromino5(self)}

    def init_world(self, event_counter):
        self.event_counter = event_counter
        self.shape_id_counter = 0
        self.occupied_cell_dict = {}
        self.current_shape_list = []
        self.shape_dict = {}
        self.turn_counter = 0

    def add_shape_to_world(self, shape_name, shape_id_counter, color):
        shape = self.master_shape_dict[shape_name]
        shape.init_shape(shape_id_counter, color)
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
                    color = 'silver'
                else:
                    if (i, j) in self.occupied_cell_dict:
                        shape_id = self.occupied_cell_dict[(i, j)]
                        color = self.shape_dict[shape_id].color
                    else:
                        color = 'grey'

                values = config.Shape.color_value_dict[color]
                r_string += "{:s},".format(str(values[0]))
                g_string += "{:s},".format(str(values[1]))
                b_string += "{:s},".format(str(values[2]))
                j += 1
            i += 1
        output_string += r_string[:-1] + "," + g_string[:-1] + "," + b_string[:-1] + '\n'
        outfile.write(output_string)
