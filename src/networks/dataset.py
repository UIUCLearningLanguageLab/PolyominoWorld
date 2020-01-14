from src import config
import random
import torch
import sys
import pickle


class DataSet:

    def __init__(self, world_state_filename, network_state_filename, features):
        self.world_state_filename = world_state_filename
        self.network_state_filename = network_state_filename
        self.feature_include_array = features

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
        self.num_included_features = None
        self.included_fv_indexes = []

        self.num_sequences = None
        self.sequence_list = None
        self.num_events = None
        self.world_size = None

        self.network_state_list = None
        self.h_size = None

        self.x = None
        self.y = None
        self.label_list = None

        self.init_dataset()

        if self.world_state_filename is not None:
            self.load_world_state_data()
        if self.network_state_filename is not None:
            self.load_network_state_data()

    def init_dataset(self):

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
            if self.feature_include_array[i] == 1:
                self.included_feature_type_list.append(feature_type)
                self.included_feature_type_index_dict[feature_type] = self.num_included_feature_types
                self.num_included_feature_types += 1
                self.included_feature_type_size_dict[feature_type] = self.feature_type_size_dict[feature_type]
                self.num_included_features += self.feature_type_size_dict[feature_type]

    def load_world_state_data(self):
        self.sequence_list = []
        self.num_events = 0
        sequence_data = []
        f = open(self.world_state_filename)
        for line in f:
            self.num_events += 1
            data = (line.strip().strip('\n').strip()).split(',')
            sequence_number = int(data[0])
            event_number = int(data[1])
            shape = data[2]
            size = int(data[3])
            color = data[4]
            variant = int(data[5])
            x_coord = int(data[6])
            y_coord = int(data[7])
            action = data[8]
            world_state = torch.tensor([float(i) for i in data[9:]])
            self.world_size = len(data[9:])

            feature_list = []
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

            feature_vector = torch.tensor(feature_list, dtype=float)
            if sequence_number != 0:
                if event_number == 0:
                    self.sequence_list.append(sequence_data)
                    sequence_data = []

            sequence_data.append([shape, size, color, variant, x_coord, y_coord, action, world_state, feature_vector])

        f.close()

        self.sequence_list.append(sequence_data)
        self.num_sequences = len(self.sequence_list)

    def load_network_state_data(self):
        f = open(self.network_state_filename, 'rb')
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
