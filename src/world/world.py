from src import config
from src.world import shapes
import sys
import random


class World:

    def __init__(self, shape_list, color_list, num_rows, num_columns):
        self.history_list = []
        self.event_counter = None
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.shapes_per_image = 1

        self.shape_id_counter = None
        self.occupied_cell_dict = None
        self.current_shape_list = None
        self.shape_dict = None
        self.turn_counter = None

        self.num_features = 0
        self.feature_list = []
        self.feature_index_dict = {}

        self.shape_list = shape_list
        self.color_list = color_list
        self.shape_list_size = len(self.shape_list)
        self.color_list_size = len(self.color_list)

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

    def generate_world(self, num_types, num_instances_per_type, num_events_per_scene):
        world_name = "w{}-{}_s{}_c{}_{}_{}_{}".format(self.num_rows, self.num_columns,
                                                      len(self.shape_list), len(self.color_list),
                                                      num_types, num_instances_per_type, num_events_per_scene)

        file_name = "data/" + world_name + ".csv"

        shape_counter = 0
        scene_counter = 0

        f = open(file_name, 'w')
        f.write(world_name + "\n")
        f.close()

        if num_types == 0:
            for i in range(9):  # num of shape types/size
                for j in range(8):  # num of colors
                    for k in range(num_instances_per_type):
                        self.init_world(scene_counter)
                        shape_name = self.shape_list[i]
                        shape_color = self.color_list[j]
                        self.add_shape_to_world(shape_name, shape_counter, shape_color)
                        self.save_world_state(file_name)
                        shape_counter += 1

                        for m in range(num_events_per_scene):
                            self.next_turn()
                            self.save_world_state(file_name)
                        scene_counter += 1
        elif num_types > 0:
            for i in range(num_types):
                for j in range(num_instances_per_type):
                    self.init_world(scene_counter)
                    shape_name = random.choice(self.shape_list)
                    shape_color = random.choice(self.color_list)
                    self.add_shape_to_world(shape_name, shape_counter, shape_color)
                    self.save_world_state(file_name)
                    shape_counter += 1

                    for k in range(num_events_per_scene):
                        self.next_turn()
                        self.save_world_state(file_name)
                    scene_counter += 1
        else:
            print("ERROR: Num Types must be >= 0")
            sys.exit()

    def add_shape_to_world(self, shape_name, shape_id_counter, color):
        shape = self.master_shape_dict[shape_name]
        shape.init_shape(shape_id_counter, color)
        self.current_shape_list.append(shape)
        self.shape_dict[shape_id_counter] = shape

    def next_turn(self):
        for i in range(self.shapes_per_image):
            self.current_shape_list[i].take_turn()
            self.turn_counter += 1

    def save_world_state(self, file_name):

        outfile = open(file_name, 'a')
        output_string = "{},{},".format(self.event_counter, self.turn_counter)

        current_shape = self.current_shape_list[0]

        for i in range(self.shapes_per_image):
            output_string += "{},{},{},{},{},{},{},".format(current_shape.name,
                                                            current_shape.size,
                                                            current_shape.color,
                                                            current_shape.current_variant,
                                                            current_shape.position[0],
                                                            current_shape.position[1],
                                                            current_shape.action_choice)

        r_string = ""
        g_string = ""
        b_string = ""

        save_list = []
        for i in range(self.num_rows):
            for j in range(self.num_columns):
                if (i, j) in self.occupied_cell_dict:
                    shape_id = self.occupied_cell_dict[(i, j)]
                    color = self.shape_dict[shape_id].color
                    save_list.append((shape_id, color))
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
