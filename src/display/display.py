import tkinter as tk
import sys
from src import config
from src.world import world
from src.networks import dataset
from tkinter import ttk
import numpy as np


class Display:

    def __init__(self, the_dataset, the_model):

        self.the_dataset = dataset.Dataset(the_dataset)
        self.i = 0  # currently active item in the dataset

        self.the_model = the_model

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

        self.i_label = tk.Label(self.button_frame, text="Current Item:", bg='white', fg='black')
        self.i_label.pack(side=tk.LEFT)

        i = tk.StringVar(self.root, value=self.i)
        self.i_entry = tk.Entry(self.button_frame, width=6, textvariable=i, relief='flat', borderwidth=0)
        self.i_entry.pack(side=tk.LEFT)

        ttk.Style().configure("TButton", padding=0, relief="flat", background="#EEEEEE", foreground='black')
        self.update_button = ttk.Button(self.button_frame, text="Update", width=8, command=self.update)
        self.update_button.pack(side=tk.LEFT)
        self.next_button = ttk.Button(self.button_frame, text="Next Item", width=8, command=self.next)
        self.next_button.pack(side=tk.LEFT)
        self.quit_button = ttk.Button(self.button_frame, text="Quit", width=8, command=sys.exit)
        self.quit_button.pack(side=tk.LEFT)

        self.draw_window()

    def draw_window(self):
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
        start_x = 250
        start_y = 20
        size = 14
        input_size = 130
        spacing = 1

        current_x = np.copy(self.the_dataset.x[self.i])

        rgb_matrix = current_x.reshape((3, self.the_dataset.num_rows, self.the_dataset.num_columns))
        for i in range(self.the_dataset.num_rows):
            for j in range(self.the_dataset.num_columns):
                r_color = self.network_hex_color(rgb_matrix[0, i, j])
                g_color = self.network_hex_color(rgb_matrix[1, i, j])
                b_color = self.network_hex_color(rgb_matrix[2, i, j])

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y,
                                                     fill=r_color, outline="#333333")

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + input_size,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + input_size,
                                                     fill=g_color, outline="#333333")

                self.network_canvas.create_rectangle(i * size + start_x,
                                                     j * size + start_y + 2*input_size,
                                                     (i + 1) * size + start_x,
                                                     (j + 1) * size + start_y + 2*input_size,
                                                     fill=b_color, outline="#333333")

    def draw_hidden_layer(self):
        pass

    def draw_output_layer(self):
        pass

    def next(self):
        if self.i < self.the_dataset.num_items:
            self.i += 1
        else:
            self.i = 0
        self.draw_window()

    def update(self):
        new_i = int(self.i_entry.get())
        if 0 <= new_i < self.the_dataset.num_items:
            self.i = new_i
            self.draw_window()

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
