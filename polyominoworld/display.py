"""
This module contains logic necessary for creating the interactive visualization in scripts/visualize_model.py
"""


import tkinter as tk
from tkinter import ttk
import sys
from typing import List

from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld import configs
from polyominoworld.helpers import FeatureVector, FeatureLabel


class Display:

    def __init__(self,
                 data: DataSet,
                 net: Network,
                 ):

        self.data = data
        self.net = net

        self.xs = [event.get_x(net.params.x_type) for event in self.data.generate_events()]

        self.x_id = 0  # currently active item in the dataset
        self.selected_unit = None

        self.square_size = configs.Display.square_size

        self.height = 900
        self.width = 1200

        self.root = tk.Tk()
        self.root.title("PolyominoWorld")

        self.network_frame = tk.Frame(self.root, height=(self.height/2)-10, width=self.width, bd=0, padx=0, pady=0)
        self.weight_frame = tk.Frame(self.root, height=(self.height/2)-10, width=self.width, bd=0, padx=0, pady=0)
        self.button_frame = tk.Frame(self.root, height=20, bg="white", width=self.width, bd=0, padx=0, pady=0)

        self.button_frame.pack()
        self.network_frame.pack()
        self.weight_frame.pack()

        self.network_canvas = tk.Canvas(self.network_frame, height=self.height/2, width=self.width,
                                        bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        self.weight_canvas = tk.Canvas(self.network_frame, height=self.height/2, width=self.width,
                                       bd=5, bg="#333333", highlightthickness=0, relief='ridge')
        self.network_canvas.pack()
        self.weight_canvas.pack()
        self.network_canvas.bind("<Button-1>", self.network_click)

        self.i_label = tk.Label(self.button_frame, text="Current Item:", bg='white', fg='black')
        self.i_label.pack(side=tk.LEFT)

        x_id = tk.StringVar(self.root, value=self.x_id)
        self.i_entry = tk.Entry(self.button_frame, width=6, textvariable=x_id, relief='flat', borderwidth=0)
        self.i_entry.pack(side=tk.LEFT)

        ttk.Style().configure("TButton", padding=0, relief="flat", background="#EEEEEE", foreground='black')
        self.update_button = ttk.Button(self.button_frame, text="Update", width=8, command=self.update)
        self.update_button.pack(side=tk.LEFT, padx=4)
        self.previous_button = ttk.Button(self.button_frame, text="Previous Item", width=12, command=self.previous)
        self.previous_button.pack(side=tk.LEFT, padx=4)
        self.next_button = ttk.Button(self.button_frame, text="Next Item", width=12, command=self.next)
        self.next_button.pack(side=tk.LEFT, padx=4)
        self.quit_button = ttk.Button(self.button_frame, text="Quit", width=8, command=sys.exit)
        self.quit_button.pack(side=tk.LEFT, padx=4)

        self.position_dict = {'World State': (20, 120, "World State"),
                              'Predicted World State': (1000, 120, "Predicted World State"),

                              'World Layer Activations': (280, 40, "Input Layer"),
                              'Predicted World Layer Activations': (800, 10, "Output Layer"),
                              'World Layer Weights': (280, 40, ""),
                              'Predicted World Layer Weights': (800, 40, ""),

                              'Predicted Feature Activations': (900, 20, "Output Layer"),
                              'Predicted Feature Weights': (900, 20, ""),

                              'Hidden Layer Activations': (520, 20, "Hidden Layer"),
                              'Hidden Layer Weights': (520, 20, "")
                              }
        self.draw_window()

    def update_current_item(self):
        self.i_entry.delete(0, tk.END)  # deletes the current value
        self.i_entry.insert(0, self.x_id)  # inserts new value assigned by 2nd parameter
        x = self.xs[self.x_id].clone()
        return x

    def draw_window(self):

        x = self.update_current_item()
        o, h = self.net.forward(x, return_h=True)

        self.network_canvas.delete("all")
        self.weight_canvas.delete("all")

        # draw the world for the current x
        self.draw_world(self.network_canvas, x, 'World State')

        # draw the input layer for the current x
        self.draw_world_layer(self.network_canvas, x, 1, 'World Layer Activations')

        # draw the hidden layer for the current x
        self.draw_hidden_layer(self.network_canvas, h, 1, 'Hidden Layer Activations')

        # draw the output layer for the current x
        if self.net.params.y_type == 'features':
            self.draw_feature_layer(self.network_canvas, o.detach().numpy(), 'Predicted Feature Activations')
        elif self.net.params.y_type == 'world':
            self.draw_world(self.network_canvas, o.detach().numpy(), 'Predicted World State')
            self.draw_world_layer(self.network_canvas, o.detach().numpy(), None, 'Predicted World Layer Activations')
        else:
            raise AttributeError("Network y_type {} not recognized".format(self.net.params.y_type))

        # if a unit is selected, draw the weights to/from the unit
        if self.selected_unit is not None:
            self.draw_weights()

    def draw_weights(self):

        # get the selected unit info
        selected_unit_info = self.selected_unit.split('_')
        if selected_unit_info[1] == 'bias':
            index = 'bias'
        else:
            index = int(selected_unit_info[1])

        # if the selected unit is an input, hidden, or output activation unit, erase canvas and draw title
        if (selected_unit_info[0] in ('i', 'o', 'h')) and (selected_unit_info[2] == 'activations'):
            self.weight_canvas.delete("all")
            self.weight_canvas.create_text(self.width / 2, 25, text="Weight Display",
                                           font="Arial 20 bold", fill='white')

        # if selected unit is an input activation, draw weights to hidden layer
        if selected_unit_info[0] == 'i' and selected_unit_info[2] == 'activations':
            self.weight_canvas.create_text(self.width / 2, 60,
                                           text="{}{}-->h weights".format(selected_unit_info[0],
                                                                          selected_unit_info[1]),
                                           font="Arial 14 bold", fill='white')
            if index == 'bias':
                weights = self.net.h_x.bias.detach().numpy()
            else:
                weight_matrix = self.net.h_x.weight.detach().numpy()
                weights = weight_matrix[:, index]
            self.draw_hidden_layer(self.weight_canvas, weights, None, 'Hidden Layer Weights')

        # if selected unit is a hidden unit, draw weights to input and output layer
        elif selected_unit_info[0] == 'h':
            if selected_unit_info[1] != 'bias':
                self.weight_canvas.create_text(320, 30,
                                               text="input-->h{} weights".format(selected_unit_info[1]),
                                               font="Arial 14 bold", fill='white')
                index = int(selected_unit_info[1])
                h_x_weight_matrix = self.net.h_x.weight.detach().numpy()
                h_x_weight_vector = h_x_weight_matrix[index, :]
                h_x_bias_vector = self.net.h_x.bias.detach().numpy()
                h_x_bias = h_x_bias_vector[index]
                y_h_weight_matrix = self.net.y_h.weight.detach().numpy()
                y_h_weight_vector = y_h_weight_matrix[:, index]
                self.draw_world_layer(self.weight_canvas, h_x_weight_vector, h_x_bias, 'World Layer Weights')
            else:
                y_h_weight_vector = self.net.y_h.bias.detach().numpy()
            if self.net.params.y_type == 'world':
                self.draw_world_layer(self.weight_canvas, y_h_weight_vector, None, 'Predicted World Layer Weights')
            elif self.net.params.y_type == 'features':
                self.draw_feature_layer(self.weight_canvas, y_h_weight_vector, 'Predicted Feature Weights')

            self.weight_canvas.create_text(900, 20,
                                           text="h{}-->output weights".format(selected_unit_info[1]),
                                           font="Arial 14 bold", fill='white')

        # if selected unit is an output unit, draw weights to hidden layer
        elif selected_unit_info[0] == 'o':
            self.weight_canvas.create_text(self.width / 2, 60,
                                           text="{}{}-->h weights".format(selected_unit_info[0],
                                                                          selected_unit_info[1]),
                                           font="Arial 14 bold", fill='white')

            weight_matrix = self.net.y_h.weight.detach().numpy()
            weights = weight_matrix[index, :]
            self.draw_hidden_layer(self.weight_canvas, weights, None, 'Hidden Layer Weights')

        self.root.update()

    def draw_world(self, canvas, x, condition):
        start_x = self.position_dict[condition][0]
        start_y = self.position_dict[condition][1]
        title = self.position_dict[condition][2]

        rgb_matrix = x.reshape((3, configs.World.num_rows, configs.World.num_cols))
        size = 20

        canvas.create_text(start_x + 40, start_y - 20,
                           text="{}".format(title),
                           font="Arial 16 bold", fill='white')

        unit_counter = 0
        for i in range(configs.World.num_rows):
            for j in range(configs.World.num_cols):
                tag = self.create_tag_name(condition, unit_counter, False)
                color = self.rgb_to_hex(rgb_matrix[0, i, j], rgb_matrix[1, i, j], rgb_matrix[2, i, j])
                canvas.create_rectangle(i * size + start_x,
                                        j * size + start_y,
                                        (i + 1) * size + start_x,
                                        (j + 1) * size + start_y,
                                        fill=color, outline=color, tag=tag)
                unit_counter += 1

    def draw_world_layer(self, canvas, layer, bias, condition):
        # get the starting x,y position
        start_x = self.position_dict[condition][0]
        start_y = self.position_dict[condition][1]
        title = self.position_dict[condition][2]

        # decide how big the units should be, based on their number
        if configs.World.num_rows > 8:
            size = 10
        else:
            size = 14

        # Write the layer name
        canvas.create_text(start_x+50, start_y-20,
                           text=title, font="Arial 20 bold", fill='white')

        color_label_list = ['Red', 'Green', 'Blue']

        # if it is an input layer, create the bias unit
        if bias is not None:
            canvas.create_text(start_x - 30, start_y+10,
                               text="Bias".format(title), font="Arial 14 bold", fill='white')
            bias_tag = self.create_tag_name(condition, None, True)
            if self.selected_unit == bias_tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'
            canvas.create_rectangle(start_x, start_y+3, start_x + size, start_y + size + 3,
                                    fill='red', outline=outline_color, tag=bias_tag)

        # draw the world layer
        unit_counter = 0
        grid_size = configs.World.num_rows * size + 10
        rgb_matrix = layer.reshape((3, configs.World.num_rows, configs.World.num_cols))
        for k in range(3):
            canvas.create_text(start_x - 30, start_y + 90 + (k*grid_size),
                               text="{}".format(color_label_list[k]).rjust(5),
                               font="Arial 14 bold", fill='white')
            for i in range(configs.World.num_rows):
                for j in range(configs.World.num_cols):
                    tag = self.create_tag_name(condition, unit_counter, False)
                    fill_color = self.network_hex_color(rgb_matrix[k, i, j])

                    if self.selected_unit == tag:
                        outline_color = 'yellow'
                    else:
                        outline_color = 'black'

                    canvas.create_rectangle(i * size + start_x,
                                            j * size + start_y + (k*grid_size) + 30,
                                            (i + 1) * size + start_x,
                                            (j + 1) * size + start_y + (k*grid_size) + 30,
                                            fill=fill_color, outline=outline_color, tag=tag)
                    unit_counter += 1

    def draw_hidden_layer(self,
                          canvas,
                          h,
                          bias,
                          condition,
                          size: int = configs.Display.hidden_layer_size,
                          spacing: int = configs.Display.hidden_layer_spacing,
                          ):
        start_x = self.position_dict[condition][0]
        start_y = self.position_dict[condition][1]
        title = self.position_dict[condition][2]

        hidden_units_per_column = 8

        if bias is not None:
            canvas.create_text(start_x - 30, start_y + 90,
                               text="Bias", font="Arial 14 bold", fill='white')
            bias_tag = self.create_tag_name(condition, None, True)
            if self.selected_unit == bias_tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'
            canvas.create_rectangle(start_x, start_y + 80, start_x + size, start_y + size + 80,
                                    fill='red', outline=outline_color, tag=bias_tag)

        canvas.create_text(start_x+80, start_y,
                           text=title, font="Arial 20 bold", fill='white')

        for i in range(len(h)):
            tag = self.create_tag_name(condition, str(i), False)
            y1 = start_y + (size + spacing) * i
            fill_color = self.network_hex_color(h[i])
            if self.selected_unit == tag:
                border_color = 'yellow'
            else:
                border_color = 'black'
            canvas.create_rectangle(start_x, y1 + 120, start_x + size, y1 + size + 120,
                                    fill=fill_color, outline=border_color, tags=tag)
            if (i+1) % hidden_units_per_column == 0:
                start_x += size + spacing
                start_y -= hidden_units_per_column * (size + spacing)

    def draw_feature_layer(self,
                           canvas,
                           layer,
                           condition,
                           size: int = configs.Display.feature_layer_size,
                           spacing: int = configs.Display.feature_layer_spacing,

                           ):
        start_x = self.position_dict[condition][0]
        start_y = self.position_dict[condition][1]
        title = self.position_dict[condition][2]

        canvas.create_text(start_x, start_y, text=title, font="Arial 20 bold", fill='white')

        soft_max = layer / layer.sum()  # TODO this is temporary -
        # softmax was previously calculated only over activations corresponding to a single feature type,
        # to properly normalize only over features within a type

        feature_labels: List[FeatureLabel] = FeatureVector.get_feature_labels()
        assert len(layer) == len(feature_labels)

        previous_feature_type_name = 'n/a'
        feature_type_id = 0

        # for each feature
        for n, feature_label in enumerate(feature_labels):

            # increment feature_type_id only when feature_type (e.g. shape, color) changes
            if feature_label.type_name != previous_feature_type_name:
                feature_type_id += 1
            previous_feature_type_name: str = feature_label.type_name

            tag = self.create_tag_name(condition, n, False)
            fill_color = self.network_hex_color(layer[n])
            if self.selected_unit == tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'

            x1 = start_x
            y1 = start_y + ((size + spacing) * n) + 20 + (feature_type_id * 10)

            canvas.create_rectangle(x1, y1, x1 + size, y1 + size,
                                    fill=fill_color, outline=outline_color, tags=tag)
            canvas.create_text(start_x - 60, y1 + 10,
                               text="{}".format(feature_label).rjust(12),
                               font="Arial 12 bold", fill='white')

            if condition == 'Predicted Feature Activations':
                value = "{:0.1f}%".format(100 * soft_max[n])
                bar_size = round(100 * soft_max[n])
                canvas.create_rectangle(x1 + 50, y1 + 4, start_x + 50 + bar_size, y1 + 16,
                                        fill='blue')
                canvas.create_text(x1 + 200, y1 + 10 + 2,
                                   text=value, font="Arial 12 bold", fill='white')

    def next(self):
        if self.x_id < len(self.data):
            self.x_id += 1
        else:
            self.x_id = 0
        self.update_current_item()
        self.draw_window()

    def previous(self):
        if self.x_id > 0:
            self.x_id -= 1
        else:
            self.x_id = len(self.data) - 1
        self.update_current_item()
        self.draw_window()

    def update(self):
        new_i = int(self.i_entry.get())
        if 0 <= new_i < len(self.data):
            self.x_id = new_i
            self.update_current_item()
            self.draw_window()

    def network_click(self, event):
        tag = self.get_tags(event)
        if tag is not None:
            if tag == self.selected_unit:
                self.selected_unit = None
                self.draw_window()
            else:
                if tag[0] == 'i':
                    self.selected_unit = tag
                    self.draw_window()
                if tag[0] == 'h':
                    self.selected_unit = tag
                    self.draw_window()
                if tag[0] == 'o':
                    self.selected_unit = tag
                    self.draw_window()

    def get_tags(self, event):
        x, y = event.x, event.y
        ids = self.network_canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        if len(ids) > 0:
            tag = self.network_canvas.itemcget(ids[0], "tags").split()[0]
        else:
            tag = None
        return tag

    @staticmethod
    def rgb_to_hex(r, g, b):
        def clamp(x):
            return max(0, min(x, 255))
        scaled_r = ((r + 1) * 128) - 1
        scaled_g = ((g + 1) * 128) - 1
        scaled_b = ((b + 1) * 128) - 1

        hex_value = "#{0:02x}{1:02x}{2:02x}".format(clamp(int(scaled_r)), clamp(int(scaled_g)), clamp(int(scaled_b)))
        return hex_value

    @staticmethod
    def network_hex_color(value):
        try:
            value = value.detach().numpy()
        except:
            pass

        abs_value = 1 - abs(value)
        scaled_value = int(round(255 * abs_value, 0))
        if scaled_value < 0:
            scaled_value = 0
        if scaled_value > 255:
            scaled_value = 255
        hex_value = hex(scaled_value)[2:]

        if len(hex_value) == 1:
            hex_value = "0" + hex_value

        if value > 0:
            return '#{}ff{}'.format(hex_value, hex_value)
        elif value < 0:
            return '#ff{}{}'.format(hex_value, hex_value)
        else:
            return "#ffffff"

    @staticmethod
    def create_tag_name(title, index, bias):
        # get the tag header and footer
        if title == 'World State':
            tag_header = "w_"
            tag_footer = '_actual'
        elif title == 'Predicted World State':
            tag_header = "w_"
            tag_footer = '_predicted'
        elif title == 'World Layer Activations':
            tag_header = "i_"
            tag_footer = '_activations'
        elif title == 'Predicted World Layer Activations':
            tag_header = 'o_'
            tag_footer = '_activations'
        elif title == 'Hidden Layer Activations':
            tag_header = 'h_'
            tag_footer = '_activations'
        elif title == 'Hidden Layer Weights':
            tag_header = 'h_'
            tag_footer = '_weights'
        elif title == 'World Layer Weights':
            tag_header = "i_"
            tag_footer = '_weights'
        elif title == 'Predicted World Layer Weights':
            tag_header = 'o_'
            tag_footer = '_weights'
        elif title == 'Predicted Feature Activations':
            tag_header = 'o_'
            tag_footer = '_activations'
        elif title == 'Predicted Feature Weights':
            tag_header = 'o_'
            tag_footer = '_weights'
        else:
            print("ERROR: Unrecognized layer type {}".format(title))
            raise RuntimeError

        if bias:
            tag = tag_header + "bias" + tag_footer
        else:
            tag = tag_header + str(index) + tag_footer

        return tag
