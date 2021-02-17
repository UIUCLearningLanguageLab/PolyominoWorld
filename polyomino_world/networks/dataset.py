from polyomino_world import config
from polyomino_world.world import world
import random
import torch
import sys
import os
import pickle
import numpy as np


class DataSet:

    def __init__(self, world_state_data, network_state_filename, features, processor):
        self.world_state_data = world_state_data
        self.network_state_filename = network_state_filename
        self.feature_include_array = features
        self.processor = processor
        self.dataset_name = None

        self.feature_type_list = []
        self.feature_type_index_dict = {}
        self.feature_type_size_dict = {}
        self.num_feature_types = 0

        self.feature_list_dict = {}
        self.feature_index_dict = {}

        self.included_feature_type_list = []
        self.included_feature_type_size_dict = {}
        self.included_feature_type_index_dict = {}
        self.num_included_feature_types = None
        self.included_feature_list = []
        self.included_feature_index_dict = {}
        self.num_included_features = None
        self.included_fv_indexes = []

        self.num_sequences = None
        self.sequence_list = None
        self.num_events = None
        self.world_size = None
        self.num_rows = None
        self.num_columns = None
        self.all_color_label_list = None
        self.all_color_rgb_matrix = None
        self.master_color_label_list = None
        self.master_shape_position_list = None
        self.master_shape_label_list = None
        # self.test_world = None
        self.master_shape_list = None

        self.network_state_list = None
        self.h_size = None

        self.x = None
        self.y = None
        self.label_list = None

        self.init_dataset()

    def init_dataset(self):
        print(self.world_state_data)
        
        # all this code through 118 is about whether the data set includes all the features (color, shape, size, action)
        if self.network_state_filename is not None:
            self.load_network_state_data()

        self.feature_type_list = config.Shape.master_feature_type_list
        self.feature_type_index_dict = config.Shape.master_feature_type_index_dict
        self.num_feature_types = config.Shape.num_master_feature_types

        self.feature_list_dict['shape'] = config.Shape.master_shape_list
        self.feature_list_dict['size'] = config.Shape.master_size_list
        self.feature_list_dict['color'] = config.Shape.master_color_list
        self.feature_list_dict['action'] = config.Shape.master_action_list

        for feature_type in self.feature_type_list:
            self.feature_type_size_dict[feature_type] = len(self.feature_list_dict[feature_type])
            self.feature_index_dict[feature_type] = {}
            num_features = len(self.feature_list_dict[feature_type])

            for i in range(num_features):
                feature = self.feature_list_dict[feature_type][i]
                self.feature_index_dict[feature_type][feature] = i

        start = 0
        for i in range(self.num_feature_types):
            feature_type = self.feature_type_list[i]
            feature_type_size = self.feature_type_size_dict[feature_type]

            if self.feature_include_array[i] == 1:
                stop = start + feature_type_size - 1
                self.included_fv_indexes.append((start, stop))
                start += feature_type_size

        self.num_included_feature_types = 0
        self.num_included_features = 0
        for i in range(self.num_feature_types):
            feature_type = self.feature_type_list[i]
            feature_list = self.feature_list_dict[feature_type]
            if self.feature_include_array[i] == 1:
                self.included_feature_type_list.append(feature_type)
                self.included_feature_type_index_dict[feature_type] = self.num_included_feature_types
                self.num_included_feature_types += 1
                self.included_feature_type_size_dict[feature_type] = self.feature_type_size_dict[feature_type]

                for j in range(len(feature_list)):
                    feature = feature_list[j]
                    self.included_feature_list.append(feature)
                    self.included_feature_index_dict[feature] = self.num_included_features
                    self.num_included_features += 1

        self.all_color_label_list = []
        self.master_color_label_list = []
        self.all_color_rgb_matrix = np.zeros([len(config.Shape.color_value_dict), 3], float)

        i = 0
        # this creates a num_colors x 3 matrix, with the RGB values for each color
        for color in config.Shape.color_value_dict:
            if color != 'grey':
                self.master_color_label_list.append(color) # this is the list of colors that will show up in the network output
            self.all_color_label_list.append(color) # all the colors we want to be able to look up RGB codes for
            rgb = config.Shape.color_value_dict[color]
            self.all_color_rgb_matrix[i, 0] = rgb[0]
            self.all_color_rgb_matrix[i, 1] = rgb[1]
            self.all_color_rgb_matrix[i, 2] = rgb[2]
            i += 1

        self.master_shape_list = config.Shape.master_shape_list

        if isinstance(self.world_state_data, world.World):
            self.parse_world_state_data(self.world_state_data.history_list)
            self.dataset_name = self.world_state_data.world_name
        
        else:
            if os.path.isfile('data/' + self.world_state_data):
                data = self.load_world_state_data()
                self.parse_world_state_data(data)
                self.dataset_name = 'data/' + self.world_state_data
            else:
                print("File {} doesn't exist".format('data/' + self.world_state_data))

        # self.master_shape_position_list = []
        # self.master_shape_label_list = []
        # current_variant_list = [[0],[0,1],[0,1],[0,1,2,3],[0],[0,1],[0,1,2,3],[0,1,2,3,4,5,6,7],[0,1,2,3]]
        # current_variant_list0 = [[0],[0],[0],[0],[0],[0],[0],[0],[1]]
        # current_variant_list1 = [[0],[0],[0],[0,1],[0],[0],[0,1],[0,1,2,3],[0,1]]
        # current_variant_list2 = [[0],[1],[1],[2,3],[0],[1],[2,3],[4,5,6,7],[2,3]]
        # current_variant_dict1 = {'monomino':[0],
        #                      'domino':[0], 
        #                      'tromino1':[0],
        #                      'tromino2':[0,1],
        #                      'tetromino1':[0], 
        #                      'tetromino2':[0], 
        #                      'tetromino3':[0,1], 
        #                      'tetromino4':[0,1,2,3], 
        #                      'tetromino5':[0,1]}

        # self.test_world = world.World(self.master_shape_list, ['black'], 4, 4, None, 1, 1, 1, None, current_variant_list)

        # shape_counter = 0
        # for i in range(len(self.master_shape_list)):  # num of shape types/size
        #     self.test_world.reset_world()
        #     shape_name = self.master_shape_list[i]
        #     shape_color = 'black'
        #     current_variant = current_variant_dict1[shape_name]
        #     self.test_world.add_shape_to_world(shape_name, shape_counter, shape_color, current_variant)
        #     current_shape = self.test_world.current_shape_list[-1]
        #     for j in range(current_shape.num_variants):
        #         active_cells = current_shape.active_cell_dict[j]
        #         self.master_shape_label_list.append(shape_name)
        #         self.master_shape_position_list.append(set(active_cells))

    def load_world_state_data(self):
        world_history_list = []
        f = open('data/' + self.world_state_data)
        for line in f:
            world_history_list.append(line.strip().strip('\n').strip())
        f.close()
        return world_history_list

    def parse_world_state_data(self, world_history_list):
        self.sequence_list = []
        self.num_events = 0
        sequence_data = []
        for line in world_history_list:
            self.num_events += 1
            data = line.split(',')
            sequence_number = int(data[0])
            event_number = int(data[1])
            shape_id = int(data[2])

            shape = data[3]
            size = int(data[4])
            color = data[5]
            variant = int(data[6])
            x_coord = int(data[7])
            y_coord = int(data[8])
            action = data[9]

            if self.processor == 'GPU':
                world_state = torch.cuda.FloatTensor([float(i) for i in data[10:]])
            else:
                data_list = []
                for datum in data[10:]:
                    data_list.append(float(datum))

                world_state = torch.tensor(data_list, dtype=float)

            self.world_size = len(data[10:])

            feature_list = []

            #print(self.feature_include_array, self.num_included_feature_types)

            for i in range(self.num_included_feature_types):
                current_feature_list = []
                feature_type = self.included_feature_type_list[i]
                num_features = self.feature_type_size_dict[feature_type]
                for j in range(num_features):
                    current_feature_list.append(0)

                if feature_type == 'shape':
                    feature_index = self.feature_index_dict['shape'][shape]
                elif feature_type == 'size':
                    feature_index = self.feature_index_dict['size'][size]
                elif feature_type == 'color':
                    feature_index = self.feature_index_dict['color'][color]
                elif feature_type == 'action':
                    feature_index = self.feature_index_dict['action'][action]
                else:
                    print("Feature Type Not Recognized")
                    sys.exit()

                current_feature_list[feature_index] = 1
                for j in range(len(current_feature_list)):
                    feature_list.append(current_feature_list[j])

            if self.processor == 'GPU':
                feature_vector = torch.cuda.FloatTensor(feature_list)
            else:
                feature_vector = torch.tensor(feature_list, dtype=float)

            if sequence_number != 0:
                if event_number == 0:
                    self.sequence_list.append(sequence_data)
                    sequence_data = []

            sequence_data.append([shape, size, color, variant, x_coord, y_coord, action, world_state, feature_vector])

        self.sequence_list.append(sequence_data)
        self.num_sequences = len(self.sequence_list)

        # todo get this information from the datafile name, or better yet redo dataset so it is a directory with
        # todo separate params file and data file, and then get from params file
        self.num_rows = int((self.world_size/3)**0.5)
        self.num_columns = int((self.world_size / 3) ** 0.5)

    def load_network_state_data(self):
        f = open("models/" + self.network_state_filename, 'rb')
        self.network_state_list = pickle.load(f)  # [[x, y, o, h], [x, y, o, h], ...]
        f.close()

        self.h_size = len(self.network_state_list[0][3])

        event_counter = 0
        for i in range(self.num_sequences):
            sequence = self.sequence_list[i]
            for j in range(len(sequence)):
                hidden_state = self.network_state_list[event_counter][3]
                self.sequence_list[i][j].append(hidden_state)
                event_counter += 1

    def create_xy(self, net, shuffle_sequences, shuffle_events):
        x = []
        y = []
        self.label_list = []

        sequence_index_list = list(range(self.num_sequences))
        if shuffle_sequences:
            random.shuffle(sequence_index_list)

        for i in range(self.num_sequences):
            sequence_index = sequence_index_list[i]
            sequence = self.sequence_list[sequence_index]
            sequence_length = len(sequence)
            event_index_list = list(range(sequence_length))
            if shuffle_events:
                random.shuffle(event_index_list)
            for j in range(sequence_length):
                event_index = event_index_list[j]
                event = sequence[event_index]

                label_list = []
                for k in range(self.num_included_feature_types):
                    feature_type = self.included_feature_type_list[k]
                    if feature_type == 'shape':
                        label_list.append(event[0])
                    elif feature_type == 'size':
                        label_list.append(event[1])
                    elif feature_type == 'color':
                        label_list.append(event[2])
                    elif feature_type == 'action':
                        label_list.append(event[6])

                self.label_list.append(label_list)

                if net.x_type == 'WorldState':
                    x.append(event[7])
                elif net.x_type == 'HiddenState':
                    x.append(event[9])
                else:
                    print("x_type {} not recognized while creating xy's".format(net.x_type))
                    sys.exit()

                if net.y_type == 'WorldState':
                    y.append(event[7])
                elif net.y_type == 'FeatureVector':
                    y.append(event[8])
                else:
                    print("y_type {} not recognized while creating xy's".format(net.y_type))
                    sys.exit()

        self.x = x
        self.y = y
