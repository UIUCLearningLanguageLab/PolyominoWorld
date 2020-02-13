import tkinter as tk
import sys
import numpy as np
from polyomino_world import config
from tkinter import ttk


class Display:

    def __init__(self, the_dataset, the_network):

        # todo make sure the properties of the dataset (rows, columns, maybe other things are same as network's

        self.the_dataset = the_dataset
        self.the_network = the_network
        self.i = 0  # currently active item in the dataset
        self.current_x = None
        self.current_y = None
        self.selected_unit = None

        self.square_size = config.Display.cell_size

        self.height = 900
        self.width = 1200

        self.root = tk.Tk()
        self.root.title("Polyomino World")

        self.network_frame = tk.Frame(self.root, height=self.height/2-10, width=self.width, bd=0, padx=0, pady=0)
        self.weight_frame = tk.Frame(self.root, height=self.height/2-10, width=self.width, bd=0, padx=0, pady=0)
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

        i = tk.StringVar(self.root, value=self.i)
        self.i_entry = tk.Entry(self.button_frame, width=6, textvariable=i, relief='flat', borderwidth=0)
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

        self.the_dataset.create_xy(self.the_network, False, False)

        self.position_dict = {'feature_output_layer': (900, 20, 'Feature Output Layer'),
                              'world_input': (20, 120, 'World Input'),
                              'world_input_layer': (280, 40, 'Input Layer'),
                              'world_output': (1000, 120, 'World Output'),
                              'world_output_layer': (800, 40, 'Output Layer')
                              }

        self.draw_window()

    def update_current_item(self):
        self.i_entry.delete(0, tk.END)  # deletes the current value
        self.i_entry.insert(0, self.i)  # inserts new value assigned by 2nd parameter
        x = self.the_dataset.x[self.i].clone()
        return x

    def calculate_network_values(self, x):
        rgb_matrix = x.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
        o, h = self.the_network.forward_item(x)
        return o, h, rgb_matrix

    def draw_window(self):

        x = self.update_current_item()
        o, h, rgb_matrix = self.calculate_network_values(x)

        self.network_canvas.delete("all")
        self.weight_canvas.delete("all")

        # draw the world for the current x
        self.draw_world('world_input', rgb_matrix)

        # draw the input layer for the current x
        self.draw_world_layer('world_input_layer', rgb_matrix)

        # draw the hidden layer for the current x
        self.draw_hidden_layer(h)

        # draw the output layer for the current x
        if self.the_network.y_type == 'FeatureVector':
            self.draw_feature_layer(o)
        elif self.the_network.y_type == 'WorldState':
            current_o = o.detach().numpy()
            rgb_matrix = current_o.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
            self.draw_world('world_output', rgb_matrix)
            self.draw_world_layer('world_output_layer', rgb_matrix)
        else:
            print("ERROR: Network y_type {} not recognized".format(self.the_network.y_type))
            sys.exit()

        if self.selected_unit is not None:
            self.draw_weights()

        self.root.update()

    def draw_world(self, layer_type, rgb_matrix):
        start_x = self.position_dict[layer_type][0]
        start_y = self.position_dict[layer_type][1]
        size = 20

        for i in range(self.the_dataset.num_rows):
            for j in range(self.the_dataset.num_columns):
                color = self.rgb_to_hex(rgb_matrix[0, i, j], rgb_matrix[1, i, j], rgb_matrix[2, i, j])
                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y,
                                                     fill=color, outline=color, tag="w{},{}".format(i, j))

    def draw_world_layer(self, layer_type, rgb_matrix):
        # get the starting x,y position
        start_x = self.position_dict[layer_type][0]
        start_y = self.position_dict[layer_type][1]

        # get the tag header
        if layer_type == 'world_input_layer':
            tag_header = "i"
        elif layer_type == 'world_output_layer':
            tag_header = 'o'
        else:
            print("ERROR: Unrecognized layer type")
            raise RuntimeError

        # decide how big the units should be, based on their number
        if self.the_dataset.num_rows > 8:
            size = 10
        else:
            size = 14

        # Write the layer name
        self.network_canvas.create_text(start_x+50, start_y-20, text=self.position_dict[layer_type][2],
                                        font="Arial 20 bold", fill='white')

        color_label_list = ['Red', 'Green', 'Blue']

        # if it is an input layer, create the bias unit
        if layer_type == 'world_input_layer':
            self.network_canvas.create_text(start_x - 30, start_y + 20,
                                            text="Bias", font="Arial 14 bold", fill='white')
            bias_tag = tag_header + "_bias"
            if self.selected_unit == bias_tag:
                outline_color = 'yellow'
            else:
                outline_color = 'black'
            self.network_canvas.create_rectangle(start_x, start_y + 13, start_x + size, start_y + size + 13,
                                                 fill='red', outline=outline_color, tag=bias_tag)

        # draw the world layer
        unit_counter = 0
        grid_spacing = 10
        grid_size = self.the_dataset.num_rows * size + grid_spacing

        for k in range(3):
            self.network_canvas.create_text(start_x - 30, start_y + 100 + (k*grid_size),
                                            text="{}".format(color_label_list[k]).rjust(5),
                                            font="Arial 14 bold", fill='white')
            for i in range(self.the_dataset.num_rows):
                for j in range(self.the_dataset.num_columns):
                    the_tag = tag_header + str(unit_counter)
                    fill_color = self.network_hex_color(rgb_matrix[k, i, j])

                    if self.selected_unit == the_tag:
                        outline_color = 'yellow'
                    else:
                        outline_color = 'black'

                    self.network_canvas.create_rectangle(i * size + start_x,
                                                         j * size + start_y + (k*grid_size) + 50,
                                                         (i + 1) * size + start_x,
                                                         (j + 1) * size + start_y + (k*grid_size) + 50,
                                                         fill=fill_color, outline=outline_color, tag=the_tag)
                    unit_counter += 1

    def draw_hidden_layer(self, h):
        start_x = 480
        start_y = 20
        size = 20
        spacing = 2
        hidden_units_per_column = 8

        self.network_canvas.create_text(start_x - 30, start_y + 90,
                                        text="Bias", font="Arial 14 bold", fill='white')
        bias_tag = "h_bias"
        if self.selected_unit == bias_tag:
            outline_color = 'yellow'
        else:
            outline_color = 'black'
        self.network_canvas.create_rectangle(start_x, start_y + 80, start_x + size, start_y + size + 80,
                                             fill='red', outline=outline_color, tag=bias_tag)

        self.network_canvas.create_text(start_x+80, start_y, text="Hidden Layer", font="Arial 20 bold", fill='white')
        for i in range(self.the_network.hidden_size):
            the_tag = "h" + str(i + 1)
            y1 = start_y + (size + spacing) * i
            fill_color = self.network_hex_color(h[i])
            if self.selected_unit == the_tag:
                border_color = 'yellow'
            else:
                border_color = 'black'
            self.network_canvas.create_rectangle(start_x, y1 + 120, start_x + size, y1 + size + 120,
                                                 fill=fill_color, outline=border_color, tags=the_tag)
            if (i+1) % hidden_units_per_column == 0:
                start_x += size + spacing
                start_y -= hidden_units_per_column * (size + spacing)

    def draw_feature_layer(self, o):
        start_x = self.position_dict['feature_output_layer'][0]
        start_y = self.position_dict['feature_output_layer'][1]

        self.network_canvas.create_text(start_x, start_y, text="Output Layer", font="Arial 20 bold", fill='white')

        size = 22
        spacing = 2

        for i in range(self.the_dataset.num_included_feature_types):
            feature_type = self.the_dataset.included_feature_type_list[i]
            included_feature_list = self.the_dataset.feature_list_dict[feature_type]
            num_features = len(included_feature_list)
            indexes = self.the_dataset.included_fv_indexes[i]
            outputs = o[indexes[0]:indexes[1]+1].clone().detach().numpy()
            soft_max = outputs / outputs.sum()

            for j in range(len(outputs)):
                the_tag = feature_type + str(j + 1)
                the_label = included_feature_list[j]
                fill_color = self.network_hex_color(outputs[j])
                if self.selected_unit == the_tag:
                    outline_color = 'yellow'
                else:
                    outline_color = 'black'

                x1 = start_x
                y1 = start_y + ((size + spacing) * i) + 20 + (j*size+spacing)

                self.network_canvas.create_rectangle(x1, y1,
                                                     x1 + size, y1 + size,
                                                     fill=fill_color, outline=outline_color, tags=the_tag)
                self.network_canvas.create_text(start_x - 60, y1 + 10 + 2, text=the_label, font="Arial 14 bold",
                                                fill='white')
                value = "{:0.1f}%".format(100 * soft_max[i])
                bar_size = round(100 * soft_max[i])
                self.network_canvas.create_rectangle(x1 + 50, y1 + 4, start_x + 50 + bar_size, y1 + 16,
                                                     fill='blue')
                self.network_canvas.create_text(x1 + 200, y1 + 10 + 2, text=value, font="Arial 12 bold",
                                                fill='white')
                num_features += 1

    def draw_weights(self):
        self.weight_canvas.delete("all")
        self.weight_canvas.create_text(self.width/2, 25, text="Weight Display".format(self.selected_unit),
                                       font="Arial 20 bold", fill='white')
        self.weight_canvas.create_text(self.width/2, 60, text="Selected Unit: {}".format(self.selected_unit),
                                       font="Arial 14 bold", fill='white')
        if self.selected_unit is not None:
            if self.selected_unit[0] == 'i':
                self.draw_to_h_weights()
            elif self.selected_unit[0] == 'h':
                self.draw_from_h_weights()
            elif self.selected_unit[0] == 'o':
                self.draw_to_h_weights()

    def draw_to_h_weights(self):
        print("Selected unit:", self.selected_unit)

        start_x = 480
        start_y = 20
        size = 20
        spacing = 2

        if self.selected_unit == 'i_bias':
            bias_tensor = self.the_network.h_x.bias
            weight_vector = bias_tensor.detach().numpy()
            self.weight_canvas.create_text(self.width / 2 - 20, start_y + 110,
                                           text="Input Bias --> Hidden Weights".format(self.selected_unit),
                                           font="Arial 12", fill='white')
        else:
            index = int(self.selected_unit[1:]) - 1
            if self.selected_unit[0] == 'i':
                h_x_tensor_matrix = self.the_network.h_x.weight
                h_x_matrix = h_x_tensor_matrix.detach().numpy()
                weight_vector = np.copy(h_x_matrix[:, index])
                self.weight_canvas.create_text(self.width / 2 - 20, start_y + 110,
                                               text="{} --> Hidden Weights".format(self.selected_unit),
                                               font="Arial 12", fill='white')
            elif self.selected_unit[0] == 'o':
                y_h_tensor_matrix = self.the_network.y_h.weight
                y_h_matrix = y_h_tensor_matrix.detach().numpy()
                weight_vector = np.copy(y_h_matrix[index, :])
                self.weight_canvas.create_text(self.width / 2 - 20, start_y + 110,
                                               text="Hidden Weights --> {}".format(self.selected_unit),
                                               font="Arial 12", fill='white')
            else:
                print("Bad Selected Unit")
                raise RuntimeError

        hidden_units_per_column = 8
        for i in range(self.the_network.hidden_size):
            y1 = start_y + (size + spacing) * i
            fill_color = self.network_hex_color(weight_vector[i])

            self.weight_canvas.create_rectangle(start_x, y1 + 120, start_x + size, y1 + size + 120,
                                                fill=fill_color, outline='black')
            if (i+1) % hidden_units_per_column == 0:
                start_x += size + spacing
                start_y -= hidden_units_per_column * (size + spacing)

    def draw_from_h_weights(self):
        print("Selected unit:", self.selected_unit)

        start_x1 = self.position_dict['world_input_layer'][0]
        start_y1 = self.position_dict['world_input_layer'][1] - 50
        start_x2 = self.position_dict['world_output_layer'][0]
        start_y2 = self.position_dict['world_output_layer'][1] - 50
        if self.the_dataset.num_rows > 8:
            size = 10
        else:
            size = 14

        if self.selected_unit == 'h_bias':
            bias_tensor = self.the_network.y_h.bias
            weight_vector = bias_tensor.detach().numpy()
            y_h_rgb_matrix = weight_vector.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
            h_x_rgb_matrix = None

        else:
            self.weight_canvas.create_text(start_x1 + 50, start_y1 + 35,
                                           text="Input-->{} Weights".format(self.selected_unit).rjust(5),
                                           font="Arial 12", fill='white')

            self.weight_canvas.create_text(start_x2 + 50, start_y2 + 35,
                                           text="{}-->Output Weights".format(self.selected_unit).rjust(5),
                                           font="Arial 12", fill='white')
            index = int(self.selected_unit[1:]) - 1
            h_x_tensor_matrix = self.the_network.h_x.weight
            h_x_matrix = h_x_tensor_matrix.detach().numpy()
            print("shape", h_x_matrix.shape)
            h_x_weight_vector = np.copy(h_x_matrix[index, :])
            h_x_rgb_matrix = h_x_weight_vector.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))

            y_h_tensor_matrix = self.the_network.y_h.weight
            y_h_matrix = y_h_tensor_matrix.detach().numpy()
            y_h_weight_vector = np.copy(y_h_matrix[:, index])
            y_h_rgb_matrix = y_h_weight_vector.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))

        # draw the world layer
        unit_counter = 0
        grid_spacing = 10
        grid_size = self.the_dataset.num_rows * size + grid_spacing

        color_label_list = ['red', 'green', 'blue']
        for k in range(3):
            if h_x_rgb_matrix is not None:
                self.weight_canvas.create_text(start_x1 - 30, start_y1 + 100 + (k * grid_size),
                                               text="{}".format(color_label_list[k]).rjust(5),
                                               font="Arial 14 bold", fill='white')

            self.weight_canvas.create_text(start_x2 - 30, start_y2 + 100 + (k * grid_size),
                                           text="{}".format(color_label_list[k]).rjust(5),
                                           font="Arial 14 bold", fill='white')

            for i in range(self.the_dataset.num_rows):
                for j in range(self.the_dataset.num_columns):
                    if h_x_rgb_matrix is not None:
                        fill_color1 = self.network_hex_color(h_x_rgb_matrix[k, i, j])
                        self.weight_canvas.create_rectangle(i * size + start_x1,
                                                            j * size + start_y1 + (k * grid_size) + 50,
                                                            (i + 1) * size + start_x1,
                                                            (j + 1) * size + start_y1 + (k * grid_size) + 50,
                                                            fill=fill_color1, outline='black')

                    fill_color2 = self.network_hex_color(y_h_rgb_matrix[k, i, j])
                    self.weight_canvas.create_rectangle(i * size + start_x2,
                                                        j * size + start_y2 + (k * grid_size) + 50,
                                                        (i + 1) * size + start_x2,
                                                        (j + 1) * size + start_y2 + (k * grid_size) + 50,
                                                        fill=fill_color2, outline='black')
                    unit_counter += 1

    def next(self):
        if self.i < self.the_dataset.num_events:
            self.i += 1
        else:
            self.i = 0
        self.update_current_item()
        self.draw_window()

    def previous(self):
        if self.i > 0:
            self.i -= 1
        else:
            self.i = self.the_dataset.num_events - 1
        self.update_current_item()
        self.draw_window()

    def update(self):
        new_i = int(self.i_entry.get())
        if 0 <= new_i < self.the_dataset.num_events:
            self.i = new_i
            self.update_current_item()
            self.draw_window()

    def network_click(self, event):
        the_tag = self.get_tags(event)
        print("Click on: ", the_tag)
        if the_tag is not None:
            if the_tag == self.selected_unit:
                self.selected_unit = None
                self.draw_window()
            else:
                if the_tag[0] == 'i':
                    self.selected_unit = the_tag
                    self.draw_window()
                if the_tag[0] == 'h':
                    self.selected_unit = the_tag
                    self.draw_window()
                if the_tag[0] == 'o':
                    self.selected_unit = the_tag
                    self.draw_window()

    def get_tags(self, event):
        x, y = event.x, event.y
        ids = self.network_canvas.find_overlapping(x - 5, y - 5, x + 5, y + 5)
        if len(ids) > 0:
            the_tag = self.network_canvas.itemcget(ids[0], "tags").split()[0]
        else:
            the_tag = None
        return the_tag

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
