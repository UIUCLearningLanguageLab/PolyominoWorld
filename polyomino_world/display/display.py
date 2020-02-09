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
        self.width = 1440

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

        self.position_dict = {'shape': (800, 60, 'Shape Outputs'),
                              'size': (800, 300, 'Size Outputs'),
                              'color': (1200, 60, 'Color Outputs'),
                              'action': (1200, 300, 'Action Outputs'),
                              'feature_output_layer': (1000, 20, 'Feature Output Layer'),
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
        self.draw_world_layer('world_input_layer', self.the_dataset.num_rows, rgb_matrix, 'input')

        # draw the hidden layer for the current x
        self.draw_hidden_layer(h)

        # draw the output layer for the current x
        if self.the_network.y_type == 'FeatureVector':
            self.draw_feature_layer(o)
        elif self.the_network.y_type == 'WorldState':
            current_o = o.detach().numpy()
            rgb_matrix = current_o.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
            self.draw_world('world_output', rgb_matrix)
            self.draw_world_layer('world_output_layer', self.the_dataset.num_rows, rgb_matrix, 'output')
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
                                                     fill=color, outline=color)

    def draw_world_layer(self, layer_type, num_rows, rgb_matrix, layer):
        start_x = self.position_dict[layer_type][0]
        start_y = self.position_dict[layer_type][1]
        size = 14

        self.network_canvas.create_text(start_x+50, start_y-20, text=self.position_dict[layer_type][2],
                                        font="Arial 20 bold", fill='white')
        self.network_canvas.create_text(start_x-20, start_y+50, text="Red", font="Arial 14 bold", fill='white')
        self.network_canvas.create_text(start_x-29, start_y+180, text="Green", font="Arial 14 bold", fill='white')
        self.network_canvas.create_text(start_x-23, start_y+310, text="Blue", font="Arial 14 bold", fill='white')

        unit_counter = 0
        grid_size = self.the_dataset.num_rows + self.the_dataset.num_columns
        for i in range(self.the_dataset.num_rows):
            for j in range(self.the_dataset.num_columns):
                if layer == 'input':
                    red_tag = "i_" + str(unit_counter)
                    green_tag = "i_" + str(unit_counter + grid_size)
                    blue_tag = "i_" + str(unit_counter + grid_size*2)
                elif layer == 'output':
                    red_tag = "o_" + str(unit_counter)
                    green_tag = "o_" + str(unit_counter + grid_size)
                    blue_tag = "o_" + str(unit_counter + grid_size*2)
                else:
                    print("ERROR: Unrecognized layer type")
                    raise RuntimeError
                unit_counter += 1

                r_color = self.network_hex_color(rgb_matrix[0, i, j])
                g_color = self.network_hex_color(rgb_matrix[1, i, j])
                b_color = self.network_hex_color(rgb_matrix[2, i, j])

                if self.selected_unit == red_tag:
                    o_color = 'yellow'
                else:
                    o_color = 'black'

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y,
                                                     fill=r_color, outline=o_color, tag=red_tag)

                if self.selected_unit == green_tag:
                    o_color = 'yellow'
                else:
                    o_color = 'black'

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + num_rows*size + 10,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + num_rows*size + 10,
                                                     fill=g_color, outline=o_color, tag=green_tag)

                if self.selected_unit == blue_tag:
                    o_color = 'yellow'
                else:
                    o_color = 'black'

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + 2*num_rows*size + 20,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + 2*num_rows*size + 20,
                                                     fill=b_color, outline=o_color, tag=blue_tag)

    def draw_hidden_layer(self, h):
        startx = 500
        starty = 40
        size = 20
        spacing = 2
        hiddens_per_column = 8

        self.network_canvas.create_text(startx+50, starty-20, text="Hidden Layer", font="Arial 20 bold", fill='white')
        for i in range(self.the_network.hidden_size):
            the_tag = "h" + str(i + 1)
            y1 = starty + (size + spacing) * i
            fcolor = self.network_hex_color(h[i])
            if self.selected_unit == the_tag:
                bcolor = 'yellow'
            else:
                bcolor = 'black'
            self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size, fill=fcolor, outline=bcolor,
                                                 tags=the_tag)
            if (i+1) % hiddens_per_column == 0:
                startx += size + spacing
                starty -= hiddens_per_column * (size + spacing)

    def draw_feature_layer(self, o):
        start_x = self.position_dict['feature_output_layer'][0]
        start_y = self.position_dict['feature_output_layer'][1]

        self.network_canvas.create_text(start_x, start_y, text="Output Layer", font="Arial 20 bold", fill='white')

        size = 22
        spacing = 2

        for i in range(self.the_dataset.num_included_feature_types):
            feature_type = self.the_dataset.included_feature_type_list[i]
            included_feature_list = self.the_dataset.feature_list_dict[feature_type]
            indexes = self.the_dataset.included_fv_indexes[i]
            outputs = o[indexes[0]:indexes[1]+1].clone().detach().numpy()
            softmax = outputs / outputs.sum()

            for i in range(len(outputs)):
                the_tag = feature_type + str(i + 1)
                the_label = included_feature_list[i]
                fcolor = self.network_hex_color(o[i])
                if self.selected_unit == the_tag:
                    bcolor = 'yellow'
                else:
                    bcolor = 'black'

                startx = self.position_dict[feature_type][0]
                starty = self.position_dict[feature_type][1]

                y1 = starty + (size + spacing) * i
                self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size,
                                                     fill=fcolor, outline=bcolor, tags=the_tag)
                self.network_canvas.create_text(startx - 60, y1 + 10 + 2, text=the_label, font="Arial 14 bold",
                                                fill='white')
                value = "{:0.1f}%".format(100 * softmax[i])
                bar_size = round(100 * softmax[i])
                self.network_canvas.create_rectangle(startx + 50, y1 + 4, startx + 50 + bar_size, y1 + 16, fill='blue')
                self.network_canvas.create_text(startx + 200, y1 + 10 + 2, text=value, font="Arial 12 bold",
                                                fill='white')

    def draw_weights(self):
        self.weight_canvas.delete("all")
        self.weight_canvas.create_text(230, 25, text="Selected Unit: {}".format(self.selected_unit),
                                       font="Arial 20 bold", fill='white')
        if self.selected_unit is not None:
            if self.selected_unit[0] == 'i':
                self.weight_canvas.create_text(100, 50, text="Input-->Hidden Weights", font="Arial 11", fill='white')
                self.draw_hx_weights()
            elif self.selected_unit[0] == 'h':
                self.weight_canvas.create_text(100, 50, text="Input-->Hidden Weights", font="Arial 11", fill='white')
                self.weight_canvas.create_text(350, 50, text="Hidden-->Output Weights", font="Arial 11", fill='white')
                self.draw_hx_weights()
                self.draw_yh_weights()
            elif self.selected_unit[0] == 'o':
                self.weight_canvas.create_text(350, 50, text="Hidden-->Output Weights", font="Arial 11", fill='white')
                self.draw_yh_weights()

    def draw_hx_weights(self):
        index = int(self.selected_unit[1:])-1

        if self.selected_unit[0] == 'i':
            startx = 90
            starty = 70
            size = 17
            spacing = 1
            weight_vector = np.copy(self.the_network.h_x[:, index])
            for i in range(len(weight_vector)):
                color = self.network_hex_color(weight_vector[i])
                x1 = startx
                y1 = starty + (size + spacing) * i
                self.weight_canvas.create_rectangle(x1, y1, x1 + size, y1 + size, fill=color)
                print(i, x1, y1, x1 + size, y1 + size)
                if i == 9:
                    print("HERE")
                    startx += size + spacing
                    starty -= 10 * (size + spacing)

        elif self.selected_unit[0] == 'h':
            startx = 30
            starty = 70
            size = 30
            spacing = 1
            weight_vector = np.copy(self.the_network.h_x[index, :])
            weight_matrix = weight_vector.reshape((5, 5))
            for i in range(weight_matrix.shape[0]):
                for j in range(weight_matrix.shape[1]):
                    color = self.network_hex_color(weight_matrix[i, j])

                    x1 = startx + (size + spacing) * i
                    y1 = starty + (size + spacing) * j
                    try:
                        self.weight_canvas.create_rectangle(x1, y1, x1 + size, y1 + size, fill=color)
                    except:
                        print(weight_matrix[i, j], color)

    def draw_yh_weights(self):
        index = int(self.selected_unit[1:])-1

        if self.selected_unit[0] == 'h':
            startx = 330
            starty = 65
            size = 18
            spacing = 1
            weight_vector = np.copy(self.the_network.o_h[:, index])
            for i in range(weight_vector.shape[0]):
                color = self.network_hex_color(weight_vector[i])
                x1 = startx
                y1 = starty + (size + spacing) * i
                self.weight_canvas.create_rectangle(x1, y1, x1 + size, y1 + size, fill=color)

        elif self.selected_unit[0] == 'o':
            startx = 330
            starty = 65
            size = 18
            spacing = 1
            weight_vector = np.copy(self.the_network.o_h[index, :])
            for i in range(weight_vector.shape[0]):
                color = self.network_hex_color(weight_vector[i])
                x1 = startx
                y1 = starty + (size + spacing) * i
                self.weight_canvas.create_rectangle(x1, y1, x1 + size, y1 + size, fill=color)
                if i == 9:
                    startx += size + spacing
                    starty -= 10 * (size + spacing)

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
