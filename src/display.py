import tkinter as tk
import sys, time
from src import config
from src import world


class Display:

    def __init__(self, the_world):

        self.the_world = the_world
        self.square_size = config.Display.cell_size
        self.running = False

        self.root = tk.Tk()
        self.root.title("Polyomino World")

        self.display_frame = tk.Frame(self.root, bd=0, padx=0, pady=0)
        self.display_frame.pack()
        self.height = (self.the_world.num_rows+2)*self.square_size
        self.width = (self.the_world.num_columns+2)*self.square_size
        self.canvas = tk.Canvas(self.display_frame, width=self.height, height=self.width, bd=0)
        self.canvas.pack()
        self.canvas.pack()

        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack()

        self.next_button = tk.Button(self.button_frame, text="Next", fg="black", command=self.next)
        self.run_button = tk.Button(self.button_frame, text="Run", fg="black", command=self.run)
        self.reset_button = tk.Button(self.button_frame, text="Reset", fg="black", command=self.reset)
        self.quit_button = tk.Button(self.button_frame, text="Quit", fg="black", command=self.quit)
        self.next_button.pack(side=tk.LEFT)
        self.run_button.pack(side=tk.LEFT)
        self.reset_button.pack(side=tk.LEFT)
        self.quit_button.pack(side=tk.LEFT)

        self.init_window()

    def init_window(self):
        self.draw_world()

    def next(self):
        self.destroy_world()
        self.draw_world()
        self.root.update()
        time.sleep(0.1)

    def run(self):
        if self.running:
            self.running = False
            self.run_button.config(text="Run")

        else:
            self.running = True
            self.run_button.config(text="Pause")

        while self.running:
            self.next()
            time.sleep(0.1)

    def destroy_world(self):
        self.canvas.delete("all")

    def draw_world(self):
        for i in range(self.the_world.num_rows + 2):
            for j in range(self.the_world.num_columns + 2):

                if i == 0 or i == self.the_world.num_rows+1 or j == 0 or j == self.the_world.num_columns+1:
                    color = 'grey'
                else:
                    if (i, j) in self.the_world.occupied_cell_dict:
                        shape_id = self.the_world.occupied_cell_dict[(i, j)]
                        color = self.the_world.shape_dict[shape_id].color
                    else:
                        color = 'silver'

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
