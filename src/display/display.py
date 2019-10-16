import tkinter as tk
import sys
from src import config
from src.world import world
from src.networks import dataset, numpy_ffnet
from tkinter import ttk
import numpy as np


class Display:

    def __init__(self, the_dataset, the_network):

        self.the_dataset = dataset.Dataset(the_dataset)
        self.i = 0  # currently active item in the dataset
        self.current_x = None
        self.current_y = None
        self.selected_unit = None

        self.the_network = numpy_ffnet.NumpyFfnet(self.the_dataset.x_size, 32, self.the_dataset.y_size, 0.001)
        self.the_network.load_weights(the_network)

        self.the_world = world.World(config.Shape.master_shape_list, config.Shape.master_color_list,
                                     self.the_dataset.num_rows, self.the_dataset.num_columns)
        self.the_world.init_world(0)
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
        self.next_button = ttk.Button(self.button_frame, text="Next Item", width=8, command=self.next)
        self.next_button.pack(side=tk.LEFT, padx=4)
        self.quit_button = ttk.Button(self.button_frame, text="Quit", width=8, command=sys.exit)
        self.quit_button.pack(side=tk.LEFT, padx=4)

        self.update_current_item()
        self.draw_network_frame()

    def update_current_item(self):
        self.i_entry.delete(0, tk.END)  # deletes the current value
        self.i_entry.insert(0, self.i)  # inserts new value assigned by 2nd parameter
        self.current_x = np.copy(self.the_dataset.x[self.i])
        self.current_y = np.copy(self.the_dataset.y[self.i])

    def draw_network_frame(self):
        self.network_canvas.delete("all")
        self.weight_canvas.delete("all")
        self.draw_world()
        self.draw_input_layer()
        self.draw_hidden_layer()
        self.draw_output_layer()

    def draw_world(self):
        start_x = 20
        start_y = 120
        size = 20

        current_x = np.copy(self.the_dataset.x[self.i])

        rgb_matrix = current_x.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
        for i in range(self.the_dataset.num_rows):
            for j in range(self.the_dataset.num_columns):
                color = self.rgb_to_hex(rgb_matrix[0, i, j], rgb_matrix[1, i, j], rgb_matrix[2, i, j])
                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y,
                                                     fill=color, outline=color)

    def draw_input_layer(self):
        start_x = 280
        start_y = 40
        size = 14
        input_size = 130

        self.network_canvas.create_text(start_x+50, start_y-20, text="Input Layer", font="Arial 20 bold", fill='white')

        self.network_canvas.create_text(start_x-20, start_y+50, text="Red", font="Arial 14 bold", fill='white')
        self.network_canvas.create_text(start_x-29, start_y+180, text="Green", font="Arial 14 bold", fill='white')
        self.network_canvas.create_text(start_x-23, start_y+310, text="Blue", font="Arial 14 bold", fill='white')

        # self.network_canvas.create_text(start_x+50, start_y+60, text="Green", font="Arial 20 bold", fill='white')
        # self.network_canvas.create_text(start_x+50, start_y+80, text="Blue", font="Arial 20 bold", fill='white')

        self.current_x = np.copy(self.the_dataset.x[self.i])

        rgb_matrix = self.current_x.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
        counter = 0
        for i in range(self.the_dataset.num_rows):
            for j in range(self.the_dataset.num_columns):
                r_color = self.network_hex_color(rgb_matrix[0, i, j])
                g_color = self.network_hex_color(rgb_matrix[1, i, j])
                b_color = self.network_hex_color(rgb_matrix[2, i, j])

                the_tag = "i" + str(counter + 1)
                if self.selected_unit == the_tag:
                    border_color = 'yellow'
                else:
                    border_color = "#333333"
                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y,
                                                     fill=r_color, outline=border_color, tags=the_tag)

                the_tag = "i" + str(counter + 1 + self.the_dataset.num_rows*self.the_dataset.num_columns)
                if self.selected_unit == the_tag:
                    border_color = 'yellow'
                else:
                    border_color = "#333333"
                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + input_size,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + input_size,
                                                     fill=g_color, outline=border_color, tags=the_tag)

                the_tag = "i" + str(counter + 1 + 2*self.the_dataset.num_rows*self.the_dataset.num_columns)
                if self.selected_unit == the_tag:
                    border_color = 'yellow'
                else:
                    border_color = "#333333"
                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + 2*input_size,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + 2*input_size,
                                                     fill=b_color, outline=border_color, tags=the_tag)
                counter += 1

    def draw_hidden_layer(self):
        startx = 500
        starty = 40
        size = 20
        spacing = 2
        hiddens_per_column = 8

        h, o = self.the_network.feedforward(self.current_x)
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

    def draw_shape_outputs(self, o):
        shape_outputs = o[:self.the_dataset.index_starts[0]]
        shape_probs = shape_outputs / shape_outputs.sum()

        startx = 800
        starty = 60
        size = 22
        spacing = 2

        for i in range(len(shape_outputs)):
            the_tag = "o" + str(i + 1)
            the_label = self.the_dataset.master_shape_list[i]
            fcolor = self.network_hex_color(o[i])
            if self.selected_unit == the_tag:
                bcolor = 'yellow'
            else:
                bcolor = 'black'

            y1 = starty + (size + spacing) * i
            self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size, fill=fcolor, outline=bcolor,
                                                 tags=the_tag)
            self.network_canvas.create_text(startx-60, y1 + 10 + 2, text=the_label, font="Arial 14 bold", fill='white')
            value = "{:0.1f}%".format(100 * shape_probs[i])
            bar_size = round(100 * shape_probs[i])
            self.network_canvas.create_rectangle(startx + 50, y1 + 4, startx + 50 + bar_size, y1 + 16, fill='blue')
            self.network_canvas.create_text(startx + 200, y1 + 10 + 2, text=value, font="Arial 12 bold", fill='white')

    def draw_size_outputs(self, o):
        size_outputs = o[self.the_dataset.index_starts[0]:self.the_dataset.index_starts[1]]
        size_probs = size_outputs / size_outputs.sum()

        startx = 800
        starty = 300
        size = 22
        spacing = 2

        for i in range(len(size_outputs)):
            the_tag = "o" + str(i + 1 + self.the_dataset.num_shapes_all)
            the_label = "size " + str(self.the_dataset.master_size_list[i])
            fcolor = self.network_hex_color(size_outputs[i])
            if self.selected_unit == the_tag:
                bcolor = 'yellow'
            else:
                bcolor = 'black'

            y1 = starty + (size + spacing) * i
            self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size, fill=fcolor, outline=bcolor,
                                                 tags=the_tag)
            self.network_canvas.create_text(startx-60, y1 + 10 + 2, text=the_label, font="Arial 14 bold", fill='white')
            value = "{:0.1f}%".format(100 * size_probs[i])
            bar_size = round(100 * size_probs[i])
            self.network_canvas.create_rectangle(startx + 50, y1 + 4, startx + 50 + bar_size, y1 + 16, fill='blue')
            self.network_canvas.create_text(startx + 200, y1 + 10 + 2, text=value, font="Arial 12 bold", fill='white')

    def draw_color_outputs(self, o):
        color_outputs = o[self.the_dataset.index_starts[1]:self.the_dataset.index_starts[2]]
        color_probs = color_outputs / color_outputs.sum()

        startx = 1200
        starty = 60
        size = 22
        spacing = 2

        for i in range(len(color_outputs)):
            the_tag = "o" + str(i + 1 + self.the_dataset.num_shapes_all + self.the_dataset.num_sizes_all)
            the_label = self.the_dataset.master_color_list[i]
            fcolor = self.network_hex_color(color_outputs[i])
            if self.selected_unit == the_tag:
                bcolor = 'yellow'
            else:
                bcolor = 'black'

            y1 = starty + (size + spacing) * i
            self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size, fill=fcolor, outline=bcolor,
                                                 tags=the_tag)
            self.network_canvas.create_text(startx-60, y1 + 10 + 2, text=the_label, font="Arial 14 bold", fill='white')
            value = "{:0.1f}%".format(100 * color_probs[i])
            bar_size = round(100 * color_probs[i])
            self.network_canvas.create_rectangle(startx + 50, y1 + 4, startx + 50 + bar_size, y1 + 16, fill='blue')
            self.network_canvas.create_text(startx + 200, y1 + 10 + 2, text=value, font="Arial 12 bold", fill='white')

    def draw_action_outputs(self, o):
        action_outputs = o[self.the_dataset.index_starts[2]:self.the_dataset.index_starts[3]]
        action_probs = action_outputs / action_outputs.sum()

        startx = 1200
        starty = 300
        size = 22
        spacing = 2

        for i in range(len(action_outputs)):
            the_tag = "o" + str(i + 1 + self.the_dataset.num_shapes_all + self.the_dataset.num_sizes_all + self.the_dataset.num_colors_all)
            the_label = self.the_dataset.master_action_list[i]
            fcolor = self.network_hex_color(action_outputs[i])
            if self.selected_unit == the_tag:
                bcolor = 'yellow'
            else:
                bcolor = 'black'

            y1 = starty + (size + spacing) * i
            self.network_canvas.create_rectangle(startx, y1, startx + size, y1 + size, fill=fcolor, outline=bcolor,
                                                 tags=the_tag)
            self.network_canvas.create_text(startx-60, y1 + 10 + 2, text=the_label, font="Arial 14 bold", fill='white')
            value = "{:0.1f}%".format(100 * action_probs[i])
            bar_size = round(100 * action_probs[i])
            self.network_canvas.create_rectangle(startx + 50, y1 + 4, startx + 50 + bar_size, y1 + 16, fill='blue')
            self.network_canvas.create_text(startx + 200, y1 + 10 + 2, text=value, font="Arial 12 bold", fill='white')

    def draw_output_layer(self):

        self.network_canvas.create_text(1000, 20, text="Output Layer", font="Arial 20 bold", fill='white')

        h, o = self.the_network.feedforward(self.the_dataset.x[self.i])
        self.draw_shape_outputs(o)
        self.draw_size_outputs(o)
        self.draw_color_outputs(o)
        self.draw_action_outputs(o)

    def next(self):
        if self.i < self.the_dataset.num_items:
            self.i += 1
        else:
            self.i = 0
        self.update_current_item()
        self.draw_network_frame()

    def update(self):
        new_i = int(self.i_entry.get())
        if 0 <= new_i < self.the_dataset.num_items:
            self.i = new_i
            self.update_current_item()
            self.draw_network_frame()

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

    def get_tags(self, event):
        x, y = event.x, event.y
        ids = self.network_canvas.find_overlapping(x, y, x, y)
        if len(ids) > 0:
            the_tag = self.network_canvas.itemcget(ids[0], "tags").split()[0]
        else:
            the_tag = None
        return the_tag

    def network_click(self, event):
        the_tag = self.get_tags(event)
        if the_tag is not None:
            if the_tag == self.selected_unit:
                self.selected_unit = None
                self.draw_network_frame()
            else:
                print(the_tag)

                if the_tag[0] == 'i':
                    self.selected_unit = the_tag
                    self.draw_network_frame()
                if the_tag[0] == 'h':
                    self.selected_unit = the_tag
                    self.draw_network_frame()
                if the_tag[0] == 'o':
                    self.selected_unit = the_tag
                    self.draw_network_frame()