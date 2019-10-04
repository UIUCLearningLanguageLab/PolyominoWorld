import numpy as np
from src import config
import random

class Dataset:

    def __init__(self, filename):
        self.filename = filename
        self.batch_size = None

        self.shape_list = config.Shape.shape_list
        self.size_list = config.Shape.size_list
        self.color_list = config.Shape.color_list
        self.action_list = config.Shape.action_list

        self.num_shapes = len(self.shape_list)
        self.num_sizes = len(self.size_list)
        self.num_colors = len(self.color_list)
        self.num_actions = len(self.action_list)

        self.x_size = (config.World.num_rows+2) * (config.World.num_columns+2) * 3
        self.y_size = self.y_size = self.num_shapes + self.num_sizes + self.num_colors + self.num_actions

        self.color_index_dict = {}
        self.shape_index_dict = {}
        self.size_index_dict = {}
        self.action_index_dict = {}

        self.event_dict = {}

        self.generate_index_dicts()
        self.load_data()

        self.num_events = len(self.event_dict)

    def generate_index_dicts(self):
        for i in range(self.num_shapes):
            self.shape_index_dict[self.shape_list[i]] = i
        for i in range(self.num_sizes):
            self.size_index_dict[self.size_list[i]] = i
        for i in range(self.num_colors):
            self.color_index_dict[self.color_list[i]] = i
        for i in range(self.num_actions):
            self.action_index_dict[self.action_list[i]] = i

    def load_data(self):
        f = open(self.filename)
        for line in f:
            data = (line.strip().strip('\n').strip()).split(',')
            event = int(data[0])
            turn = int(data[1])
            shape = data[2]
            size = int(data[3])
            color = data[4]
            variant = int(data[5])
            position = (int(data[6]), int(data[7]))
            action = data[8]
            x = np.array(data[9:], float)
            y = np.zeros([self.y_size], float)
            labels = [shape, size, color, action]

            shape_index = self.shape_index_dict[shape]
            size_index = self.size_index_dict[size] + self.num_shapes
            color_index = self.color_index_dict[color] + self.num_shapes + self.num_sizes
            action_index = self.action_index_dict[action] + self.num_shapes + self.num_sizes + self.num_colors

            y[shape_index] = 1
            y[size_index] = 1
            y[color_index] = 1
            y[action_index] = 1

            if event not in self.event_dict:
                self.event_dict[event] = []

            self.event_dict[event].append((x, y, labels, event, turn))

        f.close()

    ###########################################################################################################
    def batch_shuffled_data(self, batch_size):

        event_indexes = list(range(self.num_events))
        random.shuffle(event_indexes)
        print(event_indexes)
        x = []
        y = []
        labels = []
        events = []
        turns = []
        for index in event_indexes:
            event = self.event_dict[index]
            for state in event:
                x.append(state[0])
                y.append(state[1])
                labels.append(state[2])
                events.append(state[3])
                turns.append(state[4])
        return x, y, labels, events, turns
