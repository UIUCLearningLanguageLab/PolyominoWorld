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
from polyominoworld.helpers import FeatureVector, FeatureLabel, Event


class Display:

    def __init__(self,
                 data: DataSet,
                 net: Network,
                 ):

        self.data = data
        self.net = net
        self.net.eval()

        self.events: List[Event] = self.data.get_events()

        self.event_id = 0  # currently active item in the dataset
        self.selected_unit = None

        self.height = configs.Display.height
        self.width = configs.Display.width

        self.root = tk.Tk()
        self.root.title("PolyominoWorld")

        kw_args = {'bd': 10, 'padx': 0, 'pady': 0}

        self.network_frame = tk.Frame(self.root, height=(self.height/2)-10, width=self.width, **kw_args)
        self.weight_frame = tk.Frame(self.root, height=(self.height/2)-10, width=self.width, **kw_args)
        self.button_frame = tk.Frame(self.root, height=20, bg=configs.Display.color_bg, width=self.width, **kw_args)

        self.button_frame.pack()
        self.network_frame.pack()
        self.weight_frame.pack()

        self.canvas_net = tk.Canvas(self.network_frame, height=self.height / 2, width=self.width,
                                    bd=5, bg='#333333', highlightthickness=0, relief='ridge')
        self.canvas_weights = tk.Canvas(self.network_frame, height=self.height / 2, width=self.width,
                                        bd=5, bg="#333333", highlightthickness=0, relief='ridge')
        self.canvas_net.pack()
        self.canvas_weights.pack()
        self.canvas_net.bind("<Button-1>", self.network_click)

        self.i_label = tk.Label(self.button_frame, text="Current Item:", bg=configs.Display.color_bg_button, fg='black')
        self.i_label.pack(side=tk.LEFT)

        x_id = tk.StringVar(self.root, value=self.event_id)
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

        self.condition2position = configs.Display.condition2position
        self.draw_window()

    def update_event(self):
        self.i_entry.delete(0, tk.END)  # deletes the current value
        self.i_entry.insert(0, self.event_id)  # inserts new value assigned by 2nd parameter
        event = self.events[self.event_id]
        return event

    def draw_window(self):

        event = self.update_event()

        print()
        print(event)

        x = event.get_x(self.net.params.x_type)
        o, h = self.net.forward(x, return_h=True)
        x = x.detach().numpy()
        o = o.detach().numpy()

        self.canvas_net.delete("all")
        self.canvas_weights.delete("all")

        # draw the world for the current x
        self.draw_world(self.canvas_net, event.world_vector, 'World State')

        # draw the input layer for the current x  (separates world by color channels)
        self.draw_in_left_section(self.canvas_net, event.world_vector.as_3d(), 1, 'World Layer Activations')

        # draw the hidden layer for the current x
        self.draw_in_middle_section(self.canvas_net, h, 1, 'Hidden Layer Activations')

        # draw the output layer for the current x
        if self.net.params.y_type == 'features':
            self.draw_in_right_section(self.canvas_net, o, 'Predicted Feature Activations')
        elif self.net.params.y_type == 'world':
            self.draw_world(self.canvas_net, o, 'Predicted World State')
            self.draw_in_left_section(self.canvas_net, o, None, 'Predicted World Layer Activations')
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
            self.canvas_weights.delete("all")
            # self.canvas_weights.create_text(self.width / 2,
            #                                 25,
            #                                 text="Weight Display",
            #                                 font=configs.Display.font_xl,
            #                                 fill=configs.Display.color_text_fill)

        # if selected unit is an input activation, draw weights to hidden layer
        if selected_unit_info[0] == 'i' and selected_unit_info[2] == 'activations':
            self.canvas_weights.create_text(self.width / 2,
                                            60,
                                            text="{}{} to hidden weights".format(selected_unit_info[0],
                                                                                 selected_unit_info[1]),
                                            font=configs.Display.font_s,
                                            fill=configs.Display.color_text_fill)
            if index == 'bias':
                weights = self.net.h_x.bias.detach().numpy()
            else:
                weight_matrix = self.net.h_x.weight.detach().numpy()
                weights = weight_matrix[:, index]
            self.draw_in_middle_section(self.canvas_weights, weights, None, 'Hidden Layer Weights')

        # if selected unit is a hidden unit, draw weights to input and output layer
        elif selected_unit_info[0] == 'h':
            if selected_unit_info[1] != 'bias':
                self.canvas_weights.create_text(320,
                                                30,
                                                text="input to hidden{} weights".format(selected_unit_info[1]),
                                                font=configs.Display.font_s,
                                                fill=configs.Display.color_text_fill)
                index = int(selected_unit_info[1])
                h_x_weight_matrix = self.net.h_x.weight.detach().numpy()
                h_x_weight_vector = h_x_weight_matrix[index, :]
                h_x_bias_vector = self.net.h_x.bias.detach().numpy()
                h_x_bias = h_x_bias_vector[index]
                y_h_weight_matrix = self.net.y_h.weight.detach().numpy()
                y_h_weight_vector = y_h_weight_matrix[:, index]
                self.draw_in_left_section(self.canvas_weights, h_x_weight_vector, h_x_bias, 'World Layer Weights')
            else:
                y_h_weight_vector = self.net.y_h.bias.detach().numpy()

            if self.net.params.y_type == 'world':
                self.draw_in_left_section(self.canvas_weights, y_h_weight_vector, None, 'Predicted World Layer Weights')
            elif self.net.params.y_type == 'features':
                self.draw_in_right_section(self.canvas_weights, y_h_weight_vector, 'Predicted Feature Weights')

            self.canvas_weights.create_text(1600,
                                            30,
                                            text="h{} to output weights".format(selected_unit_info[1]),
                                            font=configs.Display.font_s,
                                            fill=configs.Display.color_text_fill)

        # if selected unit is an output unit, draw weights to hidden layer
        elif selected_unit_info[0] == 'o':
            self.canvas_weights.create_text(self.width / 2, 60,
                                            text="{}{} to hidden weights".format(selected_unit_info[0],
                                                                                 selected_unit_info[1]),
                                            font=configs.Display.font_s, fill=configs.Display.color_text_fill)

            weight_matrix = self.net.y_h.weight.detach().numpy()
            weights = weight_matrix[index, :]
            self.draw_in_middle_section(self.canvas_weights, weights, None, 'Hidden Layer Weights')

        self.root.update()

    def draw_world(self,
                   canvas,
                   world_vector,
                   condition,
                   rectangle_size: int = configs.Display.world_rectangle_size,
                   ):
        """draws the 'raw' world, with RGB values combined"""

        start_x, start_y, title = self.condition2position[condition]

        rgb_array = world_vector.as_3d()  # 3d array, (rgb, y coord, x coord)

        # canvas.create_text(start_x,
        #                    start_y,
        #                    text="{}".format(title),
        #                    font=configs.Display.font_l,
        #                    fill=configs.Display.color_text_fill)

        unit_counter = 0
        for pos_x in range(configs.World.max_y):
            for pos_y in range(configs.World.max_x):
                tag = self.create_tag_name(condition, unit_counter, False)
                color = self.rgb_to_hex(rgb_array[0, pos_x, pos_y],
                                        rgb_array[1, pos_x, pos_y],
                                        rgb_array[2, pos_x, pos_y])
                canvas.create_rectangle(pos_x * rectangle_size + start_x,
                                        pos_y * rectangle_size + start_y,
                                        (pos_x + 1) * rectangle_size + start_x,
                                        (pos_y + 1) * rectangle_size + start_y,
                                        fill=color,
                                        outline=color,
                                        tag=tag)
                unit_counter += 1

    def draw_in_left_section(self,
                             canvas,
                             layer,
                             bias,
                             condition,
                             rectangle_size: int = configs.Display.left_section_rectangle_size,
                             grid_size=configs.Display.world_grid_size,
                             ):
        """draw the world separated by RGB color channel, or weights attached to the world"""

        # get the x,y position in the display
        start_x, start_y, title = self.condition2position[condition]
        offset_top = 100

        print(f'Drawing in left section: {condition}')

        # title
        canvas.create_text(start_x,
                           start_y,
                           text=title,
                           font=configs.Display.font_xl,
                           fill=configs.Display.color_text_fill)

        color_label_list = ['Red', 'Green', 'Blue']

        # if it is an input layer, create the bias unit
        if bias is not None:
            canvas.create_text(start_x - 60,
                               start_y + offset_top - 10,
                               text="Bias".rjust(5),
                               font=configs.Display.font_s,
                               fill=configs.Display.color_text_fill)
            bias_tag = self.create_tag_name(condition, None, True)
            if self.selected_unit == bias_tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'
            canvas.create_rectangle(start_x,
                                    start_y + offset_top - 20,
                                    start_x + rectangle_size,
                                    start_y + rectangle_size + offset_top - 20,
                                    fill='red',
                                    outline=outline_color,
                                    tag=bias_tag)

        if layer.ndim != 3:
            rgb_array = layer.reshape((3, configs.World.max_x, configs.World.max_y))
        else:
            rgb_array = layer  # when visualizing each color channel, the world vector is already reshaped to 3d

        # draw world layers split by color
        unit_counter = 0
        for k in range(3):
            # write color label
            canvas.create_text(start_x - 60,
                               start_y + 80 + (k*grid_size) + offset_top,
                               text="{}".format(color_label_list[k]).rjust(5),
                               font=configs.Display.font_s,
                               fill=configs.Display.color_text_fill)

            for pos_x in range(configs.World.max_x):
                for pos_j in range(configs.World.max_y):
                    tag = self.create_tag_name(condition, unit_counter, False)
                    fill_color = self.network_hex_color(rgb_array[k, pos_x, pos_j])

                    if self.selected_unit == tag:
                        outline_color = 'yellow'
                    else:
                        outline_color = 'black'

                    canvas.create_rectangle(pos_x * rectangle_size + start_x,
                                            pos_j * rectangle_size + start_y + (k * grid_size) + offset_top,
                                            (pos_x + 1) * rectangle_size + start_x,
                                            (pos_j + 1) * rectangle_size + start_y + (k * grid_size) + offset_top,
                                            fill=fill_color, outline=outline_color, tag=tag)
                    unit_counter += 1

    def draw_in_middle_section(self,
                               canvas,
                               h,
                               bias,
                               condition,
                               rectangle_size: int = configs.Display.middle_section_rectangle_size,
                               spacing: int = configs.Display.middle_section_spacing,
                               ):
        """draw hidden activations or hidden weights"""

        start_x, start_y, title = self.condition2position[condition]

        print(f'Drawing in middle section: {condition}')

        canvas.create_text(start_x,
                           start_y,
                           text=title,
                           font=configs.Display.font_xl,
                           fill=configs.Display.color_text_fill)

        hidden_units_per_column = 8

        print(f'bias={bias}')

        if bias is not None:  # hidden layer bias
            canvas.create_text(start_x - 40,
                               start_y + 90,
                               text="Bias",
                               font=configs.Display.font_s,
                               fill=configs.Display.color_text_fill)
            bias_tag = self.create_tag_name(condition, None, True)
            if self.selected_unit == bias_tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'
            canvas.create_rectangle(start_x,
                                    start_y + 80,
                                    start_x + rectangle_size,
                                    start_y + rectangle_size + 80,
                                    fill='red',
                                    outline=outline_color,
                                    tag=bias_tag)

        for i in range(len(h)):
            tag = self.create_tag_name(condition, str(i), False)
            y1 = start_y + (rectangle_size + spacing) * i
            fill_color = self.network_hex_color(h[i])
            if self.selected_unit == tag:
                border_color = 'yellow'
            else:
                border_color = 'black'
            canvas.create_rectangle(start_x,
                                    y1 + 120,
                                    start_x + rectangle_size,
                                    y1 + rectangle_size + 120,
                                    fill=fill_color,
                                    outline=border_color,
                                    tags=tag)
            if (i+1) % hidden_units_per_column == 0:
                start_x += rectangle_size + spacing
                start_y -= hidden_units_per_column * (rectangle_size + spacing)

    def draw_in_right_section(self,
                              canvas,
                              layer,
                              condition,
                              rectangle_size: int = configs.Display.right_section_rectangle_size,
                              spacing: int = configs.Display.right_section_spacing,
                              ):
        """draw output activations"""

        start_x, start_y, title = self.condition2position[condition]

        print(f'Drawing in right section: {condition}')

        canvas.create_text(start_x,
                           start_y,
                           text=title,
                           font=configs.Display.font_xl,
                           fill=configs.Display.color_text_fill)

        logits = layer - layer.mean()   # TODO how to handle logits best for visualisation?
        # softmax was previously calculated only over activations corresponding to a single feature type,
        # to properly normalize over features within a type.
        # but this distorts the relative differences in the original logits.
        # for now, plot logits as-is, but prevent negative activations, to de-clutter display.

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
            y1 = start_y + ((rectangle_size + spacing) * n) + 20 + (feature_type_id * 10)

            # square with color proportional to logit
            canvas.create_rectangle(x1,
                                    y1,
                                    x1 + rectangle_size,
                                    y1 + rectangle_size,
                                    fill=fill_color,
                                    outline=outline_color,
                                    tags=tag)
            # feature labels
            canvas.create_text(x1 - 150,
                               y1 + 10,
                               text="{}".format(feature_label),
                               font=configs.Display.font_xs,
                               fill=configs.Display.color_text_fill)
            # bars with length proportional to logit
            if condition == 'Predicted Feature Activations':
                bar_size = round(logits[n])
                xl = x1 + 100
                yt = y1 + 10
                canvas.create_rectangle(xl,
                                        yt,
                                        xl + bar_size,
                                        yt + rectangle_size - 10 * 2,  # y offset from top
                                        fill='blue')
                # numeric label for bar
                canvas.create_text(x1 + 200,
                                   yt + 2,  # y offset from top
                                   text="{:0.1f}".format(logits[n]),
                                   font=configs.Display.font_xs,
                                   fill=configs.Display.color_text_fill)

    def next(self):
        if self.event_id < len(self.data):
            self.event_id += 1
        else:
            self.event_id = 0
        self.update_event()
        self.draw_window()

    def previous(self):
        if self.event_id > 0:
            self.event_id -= 1
        else:
            self.event_id = len(self.data) - 1
        self.update_event()
        self.draw_window()

    def update(self):
        new_i = int(self.i_entry.get())
        if 0 <= new_i < len(self.data):
            self.event_id = new_i
            self.update_event()
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
        ids = self.canvas_net.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        print('ids=', ids)
        if len(ids) > 0:
            tags = self.canvas_net.itemcget(ids[0], "tags").split()
            print('tags=', tags)
            tag = tags[0]
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
            raise RuntimeError("ERROR: Unrecognized layer type {}".format(title))

        if bias:
            tag = tag_header + "bias" + tag_footer
        else:
            tag = tag_header + str(index) + tag_footer

        return tag
