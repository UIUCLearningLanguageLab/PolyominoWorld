import tkinter as tk
import sys, time
from src import config
from src.world import world


class Display:

    def __init__(self, the_dataset, the_model):

        self.the_dataset = the_dataset
        self.the_model = the_model

        self.the_world = world.World()
        self.square_size = config.Display.cell_size

        self.root = tk.Tk()
        self.root.title("Polyomino World")

        self.display_frame = tk.Frame(self.root, bd=0, padx=0, pady=0)
        self.display_frame.pack()
        self.height = (self.the_world.num_rows)*self.square_size
        self.width = (self.the_world.num_columns)*self.square_size
        self.canvas = tk.Canvas(self.display_frame, width=self.height, height=self.width, bd=0)
        self.canvas.pack()
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.quit_button = tk.Button(self.button_frame, text="Quit", fg="black", command=self.quit)
        self.quit_button.pack(side=tk.LEFT)

        self.the_world.init_world(0)
        self.init_window()

    def init_window(self):
        self.draw_world()

    def destroy_world(self):
        self.canvas.delete("all")

    def draw_world(self):
        for i in range(self.the_world.num_rows):
            for j in range(self.the_world.num_columns):

                if (i, j) in self.the_world.occupied_cell_dict:
                    shape_id = self.the_world.occupied_cell_dict[(i, j)]
                    color = self.the_world.shape_dict[shape_id].color
                else:
                    color = config.World.background_color

                square = self.canvas.create_rectangle(i * self.square_size,
                                                      j * self.square_size,
                                                      (i + 1) * self.square_size,
                                                      (j + 1) * self.square_size,
                                                      fill=color, outline=color)
                j += 1

            i += 1

    def reset(self):
        self.the_world = world.World()
        self.destroy_world()
        self.draw_world()

    @staticmethod
    def quit():
        sys.exit()
