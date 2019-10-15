import numpy as np
from src import config
import random


class Dataset:

    def __init__(self, filename):
        self.filename = filename
        self.name = None

        self.num_rows = None
        self.num_columns = None
        self.x = None
        self.y = None
        self.label_list = None

        self.master_shape_list = None
        self.master_size_list = None
        self.master_color_list = None
        self.master_action_list = None
        self.num_shapes_all = None
        self.num_sizes_all = None
        self.num_colors_all = None
        self.num_actions_all = None
        self.index_starts = None

        self.x_size = None
        self.y_size = None
        self.num_scenes = None

        self.color_index_dict = None
        self.shape_index_dict = None
        self.size_index_dict = None
        self.action_index_dict = None

        self.scene_dict = None

        self.init_dataset()
        self.load_data()
        self.create_xy(False)

    def get_header_data(self):
        f = open(self.filename)
        line_counter = 0
        line = None
        data = None
        for line in f:
            if line_counter == 0:
                line = line.strip().strip('\n').strip()
                self.name = line
                data = line.split('_')
            else:
                break
            line_counter += 1
        f.close()

        size_data = (data[0][1:]).split('-')
        self.num_rows = int(size_data[0])
        self.num_columns = int(size_data[1])

    def init_dataset(self):
        self.get_header_data()

        self.master_shape_list = config.Shape.master_shape_list
        self.master_size_list = config.Shape.master_size_list
        self.master_color_list = config.Shape.master_color_list
        self.master_action_list = config.Shape.master_action_list

        self.num_shapes_all = len(self.master_shape_list)
        self.num_sizes_all = len(self.master_size_list)
        self.num_colors_all = len(self.master_color_list)
        self.num_actions_all = len(self.master_action_list)

        self.index_starts = [self.num_shapes_all,
                             self.num_shapes_all + self.num_sizes_all,
                             self.num_shapes_all + self.num_sizes_all + self.num_colors_all,
                             self.num_shapes_all + self.num_sizes_all + self.num_colors_all + self.num_actions_all]

        self.color_index_dict = {}
        self.shape_index_dict = {}
        self.size_index_dict = {}
        self.action_index_dict = {}

        for i in range(self.num_shapes_all):
            self.shape_index_dict[self.master_shape_list[i]] = i
        for i in range(self.num_sizes_all):
            self.size_index_dict[self.master_size_list[i]] = i
        for i in range(self.num_colors_all):
            self.color_index_dict[self.master_color_list[i]] = i
        for i in range(self.num_actions_all):
            self.action_index_dict[self.master_action_list[i]] = i

        self.x_size = self.num_rows * self.num_columns * 3
        self.y_size = self.num_shapes_all + self.num_sizes_all + self.num_colors_all + self.num_actions_all

    def load_data(self):
        self.scene_dict = {}
        line_counter = 0
        f = open(self.filename)
        for line in f:
            if line_counter > 0:
                data = (line.strip().strip('\n').strip()).split(',')
                scene = int(data[0])
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
                size_index = self.size_index_dict[size] + self.num_shapes_all
                color_index = self.color_index_dict[color] + self.num_shapes_all + self.num_sizes_all
                action_index = self.action_index_dict[action] + self.num_shapes_all + self.num_sizes_all + self.num_colors_all

                y[shape_index] = 1
                y[size_index] = 1
                y[color_index] = 1
                y[action_index] = 1

                if scene not in self.scene_dict:
                    self.scene_dict[scene] = []

                self.scene_dict[scene].append((turn, labels, x, y))

            line_counter += 1

        f.close()

        self.num_scenes = len(self.scene_dict)

    def create_xy(self, shuffle):
        x = []
        y = []
        self.label_list = []
        index_list = list(range(self.num_scenes))
        if shuffle:
            random.shuffle(index_list)
        for i in range(self.num_scenes):
            index = index_list[i]
            scene = self.scene_dict[index]
            for event in scene:
                self.label_list.append(event[1])
                x.append(event[2])
                y.append(event[3])
        self.x = np.array(x, float)
        self.y = np.array(y, float)

        self.num_items = len(self.x)