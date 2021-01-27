from polyomino_world import config
from polyomino_world.world import shapes
import sys
import random


class World:

    def __init__(self, shape_list, color_list,
                 num_rows, num_columns, custom_bounds,
                 num_types, num_sequences_per_type, num_events_per_sequence,
                 background_color, custom_variant_list):

        self.shape_list = shape_list
        self.color_list = color_list
        self.shape_list_size = len(self.shape_list)
        self.color_list_size = len(self.color_list)
        self.custom_variant_list = custom_variant_list

        self.num_rows = num_rows
        self.num_columns = num_columns
        self.custom_bounds = custom_bounds

        self.num_types = num_types
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

        if self.num_types == 0:
            self.world_name = "w{}-{}_s{}_c{}_{}_{}_{}".format(self.num_rows, self.num_columns,
                                                               len(self.shape_list), len(self.color_list),
                                                               self.num_types, self.num_sequences_per_type,
                                                               self.num_events_per_sequence)
        else:
            self.world_name = "w{}-{}_s{}_c{}_{}_{}_{}".format(self.num_rows, self.num_columns,
                                                               self.num_types, self.num_types,
                                                               self.num_types, self.num_sequences_per_type,
                                                               self.num_events_per_sequence)

        self.file_name = "data/" + self.world_name + ".csv"
        outfile = open(self.file_name, 'w')
        outfile.close()

        self.sequence_counter = 0

        self.reset_world()

    def reset_world(self):
        self.occupied_cell_dict = {}
        self.current_shape_list = []
        self.shape_dict = {}

    def generate_world(self):

        shape_counter = 0

        if self.num_types == 0:
            for i in range(len(self.shape_list)):  # num of shape types/size
                for j in range(len(self.color_list)):  # num of colors

                    for k in range(self.num_sequences_per_type):

                        self.reset_world()
                        self.event_counter = 0

                        if self.background_color == 'random':
                            self.current_background_color = (random.uniform(-1, 1),
                                                             random.uniform(-1, 1),
                                                             random.uniform(-1, 1))
                        else:
                            if self.background_color in config.Shape.color_value_dict:
                                self.current_background_color = config.Shape.color_value_dict[self.background_color]
                            else:
                                print("Background Color {} not in config.Shape.color_value_dict".format(self.background_color))
                                raise RuntimeError

                        shape_name = self.shape_list[i]
                        shape_color = self.color_list[j]

                        unique_shape_id = i # shape_counter % len(self.shape_list)
                        # print("unique_shape_id: ",unique_shape_id,"shape_counter: ", shape_counter, "\n")
                        current_custom_variant_list = self.custom_variant_list[unique_shape_id]

                        print("OCD in generate_world", self.occupied_cell_dict)
                        self.add_shape_to_world(shape_name, shape_counter, shape_color, current_custom_variant_list)

                        self.save_world_state(self.file_name)
                        shape_counter += 1

                        for m in range(self.num_events_per_sequence):
                            self.next_turn()
                            self.save_world_state(self.file_name)
                            self.event_counter += 1

                        self.sequence_counter += 1

        elif self.num_types > 0:
            for i in range(self.num_types):
                for j in range(self.num_sequences_per_type):
                    self.init_world()
                    shape_name = random.choice(self.shape_list)
                    shape_color = random.choice(self.color_list)
                    self.add_shape_to_world(shape_name, shape_counter, shape_color)
                    self.save_world_state(self.file_name)
                    shape_counter += 1

                    for k in range(self.num_events_per_sequence):
                        self.next_turn()
                        self.save_world_state(self.file_name)

        else:
            print("ERROR: Num Types must be >= 0")
            sys.exit()


    def check_location_empty(self, active_world_cells):
        all_cells_empty = True
        for i in range(len(active_world_cells)):
            if active_world_cells[i] in self.occupied_cell_dict:
                all_cells_empty = False
        return all_cells_empty


    def check_legal_position(self, active_world_cell_list, test_id_number):
        legal_position = True
        for cell in active_world_cell_list:
            if (cell[0] < 0) or (cell[1] < 0) or (cell[0] > self.num_columns-1) or (cell[1] > self.num_rows-1):
                legal_position = False
            if self.custom_bounds:
                if(cell[0] < self.custom_bounds[0]) or (cell[1] < self.custom_bounds[2]) or (cell[0] > self.custom_bounds[1]) or (cell[1] > self.custom_bounds[3]):
                    legal_position = False
            if cell in self.occupied_cell_dict:
                existing_shape_id = self.occupied_cell_dict[cell]
                if existing_shape_id != test_id_number:
                    legal_position = False
        return legal_position

    def add_shape_to_world(self, shape_name, shape_id_counter, color, current_custom_variant_list):
        shape_placed = False
        shape = self.master_shape_dict[shape_name] # may be not here
        #print("OCD Before while loop", self.occupied_cell_dict)

        try_counter = 0
        while not shape_placed:

            shape = self.master_shape_dict[shape_name]
            shape.variant_list = current_custom_variant_list
            position, active_world_cell_list = shape.init_shape(shape_id_counter, color)
            all_cells_empty = self.check_location_empty(active_world_cell_list)
            legal_position = self.check_legal_position(active_world_cell_list, shape.id_number)

            if all_cells_empty:
                if legal_position:
                    shape.position = position
                    shape.active_world_cell_list = active_world_cell_list
                    for cell in active_world_cell_list:
                        self.occupied_cell_dict[cell] = shape.id_number
                    self.current_shape_list.append(shape)
                    self.shape_dict[shape_id_counter] = shape
                    shape_placed = True
                else:
                    print("Failed to place shape due to illegal position")
            else:
                print("Failed to place shape due occupied position")

            try_counter += 1

            if try_counter > 100:
                print("Failed to place after 100 tries")
                break

            #print("OCD in while loop", self.occupied_cell_dict)
   
        # print("in add_shape_to_world: id:{}   name: {}   varaint: {}  variant_list: {}  position: {}\n".format(shape_id_counter, shape.name, shape.current_variant, shape.variant_list, shape.position))


    def next_turn(self):
        for i in range(self.shapes_per_image):
            self.current_shape_list[i].take_turn()
            # print("in next_turn: id: {}  name: {}  variant: {}\n".format(self.current_shape_list[i].id_number, self.current_shape_list[i].name, self.current_shape_list[i].current_variant))

    def save_world_state(self, file_name):
        self.test_counter += 1

        outfile = open(file_name, 'a')
        output_string = "{},{},".format(self.sequence_counter, self.event_counter)

        current_shape = self.current_shape_list[0]
        # print("in save_world_state: id:{}   name: {}   varaint: {}  variant_list: {}  position: {}\n".format(current_shape.id_number, current_shape.name, current_shape.current_variant, current_shape.variant_list, current_shape.position))
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
        # i dont know why this didnt update
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
        outfile.write(output_string)
