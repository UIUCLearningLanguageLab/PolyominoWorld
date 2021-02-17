from polyomino_world import config
from polyomino_world.world import shapes
import sys
import random


class World:

    def __init__(self, shape_list, color_list,
                 num_rows, num_columns, custom_bounds,
                 location_type, num_sequences_per_type, num_events_per_sequence,
                 background_color, custom_variant_list, name, random_seed_list):

        self.shape_list = shape_list
        self.color_list = color_list
        self.shape_list_size = len(self.shape_list)
        self.color_list_size = len(self.color_list)
        self.custom_variant_list = custom_variant_list
        self.name = name
        self.random_seed_list = random_seed_list

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.custom_bounds = custom_bounds

        self.location_type = location_type
        self.num_sequences_per_type = num_sequences_per_type
        self.num_events_per_sequence = num_events_per_sequence

        self.background_color = background_color
        self.current_background_color = None

        self.world_name = None
        self.file_name = None
        self.history_list = []
        self.sequence_counter = None
        self.event_counter = None

        self.shapes_per_image = 1

        self.shape_id_counter = None
        self.occupied_cell_dict = None
        self.current_shape_list = None
        self.shape_dict = None

        self.num_features = 0
        self.feature_list = []
        self.feature_index_dict = {}
        self.test_counter = 0

        self.master_shape_dict = {'monomino': shapes.Monomino(self),
                                  'domino': shapes.Domino(self),
                                  'tromino1': shapes.Tromino1(self),
                                  'tromino2': shapes.Tromino2(self),
                                  'tetromino1': shapes.Tetromino1(self),
                                  'tetromino2': shapes.Tetromino2(self),
                                  'tetromino3': shapes.Tetromino3(self),
                                  'tetromino4': shapes.Tetromino4(self),
                                  'tetromino5': shapes.Tetromino5(self)}

        self.init_world()

    def init_world(self):

        self.sequence_counter = 0

        if self.background_color == 'random':
            self.current_background_color = (random.uniform(-1, 1),
                                             random.uniform(-1, 1),
                                             random.uniform(-1, 1))
        else:
            if self.background_color in config.Shape.color_value_dict:
                self.current_background_color = config.Shape.color_value_dict[self.background_color]
            else:
                print("Color {} not in config.Shape.color_value_dict".format(self.background_color))
                raise RuntimeError

        self.reset_world()

    def reset_world(self):
        self.occupied_cell_dict = {}
        self.current_shape_list = []
        self.shape_dict = {}

    def generate_world(self):

        shape_counter = 0

        if self.location_type == "random":
            random_idx = 0
            for i in range(len(self.shape_list)):  # num of shape types/size 9
                for j in range(len(self.color_list)):  # num of colors 8
                    for k in range(self.num_sequences_per_type): # 10
                        shape_placed = False
                        
                        try_counter = 0

                        while not shape_placed:

                            self.reset_world()
                            self.event_counter = 0

                            shape_name = self.shape_list[i]
                            shape = self.master_shape_dict[shape_name]

                            shape_color = self.color_list[j]
                            random.seed(self.random_seed_list[random_idx])
                            variant = random.choice(self.custom_variant_list[i])
                            random_idx += 1
                            shape.init_shape(shape_counter, shape_color, variant)
                            position = self.choose_random_start_position(shape.dimensions, random_idx, random_idx+1)
                            random_idx += 2
                            shape.set_start_position(position)

                            if self.check_if_position_legal(shape.active_world_cell_list):
                                self.add_shape_to_world(shape)
                                self.save_world_state()
                                shape_counter += 1
                                shape_placed = True

                            try_counter += 1

                            if try_counter > 100:
                                print("Failed to place after 100 tries")
                                break
                        
                        if shape_placed:
                            for m in range(self.num_events_per_sequence):
                                self.next_turn()
                                self.save_world_state()
                                self.event_counter += 1

                            self.sequence_counter += 1

        elif self.location_type == "all":
            for i in range(len(self.color_list)):  # num of colors
                shape_color = self.color_list[i]
                for j in range(len(self.shape_list)):  # num of shape types/size
                    shape_name = self.shape_list[j]
                    current_custom_variant_list = self.custom_variant_list[j]
                    for k in range(len(current_custom_variant_list)):
                        variant = current_custom_variant_list[k]
                        for x in range(self.num_rows):
                            for y in range(self.num_columns):
                                self.reset_world()
                                self.event_counter = 0
                                position = (x, y)
                                shape = self.master_shape_dict[shape_name]
                                shape.init_shape(shape_counter, shape_color, variant)
                                shape.set_start_position(position)
                                if self.check_if_position_legal(shape.active_world_cell_list):
                                    self.add_shape_to_world(shape)
                                    self.save_world_state()
                                    shape_counter += 1

        else:
            print("ERROR: Num Types must be >= 0")
            sys.exit()

    def choose_random_start_position(self, dimensions, seed1, seed2):
        position = []
        if self.custom_bounds is not None:
            random.seed(seed1)
            position.append(random.randint(self.custom_bounds[0],
                                       min(self.num_columns-dimensions[0], self.custom_bounds[1])))
            random.seed(seed2)
            position.append(random.randint(self.custom_bounds[2],
                                       min(self.num_rows-dimensions[1], self.custom_bounds[3])))
        else:
            random.seed(seed1)
            position.append(random.randint(0, self.num_columns-dimensions[0]))
            random.seed(seed2)
            position.append(random.randint(0, self.num_rows-dimensions[1]))

        return position

    def check_if_position_legal(self, active_world_cell_list):
        legal_position = True
        for cell in active_world_cell_list:
            if (cell[0] < 0) or (cell[1] < 0) or (cell[0] > self.num_columns-1) or (cell[1] > self.num_rows-1):
                legal_position = False
            if self.custom_bounds:
                if(cell[0] < self.custom_bounds[0]) or \
                  (cell[1] < self.custom_bounds[2]) or \
                  (cell[0] > self.custom_bounds[1]) or \
                  (cell[1] > self.custom_bounds[3]):
                    legal_position = False
            if cell in self.occupied_cell_dict:
                legal_position = False
        return legal_position

    def add_shape_to_world(self, current_shape):
        for cell in current_shape.active_world_cell_list:
            self.occupied_cell_dict[cell] = current_shape.id_number
            self.current_shape_list.append(current_shape)
            self.shape_dict[current_shape.id_number] = current_shape

    def next_turn(self):
        for i in range(self.shapes_per_image):
            self.current_shape_list[i].take_turn()

    def save_world_state(self):

        self.test_counter += 1
        output_string = "{},{},".format(self.sequence_counter, self.event_counter)

        current_shape = self.current_shape_list[0]
        for i in range(self.shapes_per_image):
            output_string += "{},{},{},{},{},{},{},{},".format(current_shape.id_number,
                                                               current_shape.name,
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
                    values = config.Shape.color_value_dict[color]
                    save_list.append((shape_id, color))
                else:
                    values = self.current_background_color

                r_string += "{:s},".format(str(values[0]))
                g_string += "{:s},".format(str(values[1]))
                b_string += "{:s},".format(str(values[2]))

                j += 1
            i += 1

        output_string += r_string[:-1] + "," + g_string[:-1] + "," + b_string[:-1] + '\n'
        self.history_list.append(output_string)

    def print_world_history(self):
        self.world_name = "w{}-{}_s{}_c{}_location-{}_{}_{}".format(self.num_rows, self.num_columns,
                                                            len(self.shape_list), len(self.color_list),
                                                            self.location_type, self.num_sequences_per_type,
                                                            self.num_events_per_sequence)

        if self.name is not None:
            self.world_name += '_' + self.name

        self.file_name = "data/" + self.world_name + ".csv"

        # TODO check to make sure this file doesnt exist already, if it does, print an error and quit

        outfile = open(self.file_name, 'w')
        for entry in self.history_list:
            outfile.write(entry)
        outfile.close()
