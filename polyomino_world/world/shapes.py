import random
import numpy as np
from polyomino_world import config


#################################################################
class Shape:

    def __init__(self, the_world):
        self.custom_bounds = the_world.custom_bounds
        self.the_world = the_world
        self.id_number = None
        self.name = None
        self.size = None
        self.dimensions = None
        self.num_variants = None
        self.variant_list = None
        self.current_variant = None
        self.active_cell_dict = None
        self.active_cell_list = None
        self.active_world_cell_list = None
        self.color = None
        self.position = None

        self.action_list = config.Shape.master_action_list
        self.action_choice = self.action_list[0]
        self.action_probs = None
        self.flip_dict = None
        self.rotation_dict = None

    def init_shape(self, id_number, color):
        self.id_number = id_number
        self.color = color
        # print("shape name: {}    variant_list: {}\n".format(self.name, self.variant_list))
        self.current_variant = random.choice(self.variant_list)
        # print("current_variant: {}\n".format(self.current_variant))
        self.active_cell_list = self.active_cell_dict[self.current_variant]
        self.get_dimensions()
        self.action_probs = np.array(config.Shape.action_prob_list)
        position, active_world_cells = self.determine_initial_position()
        return position, active_world_cells

    def determine_initial_position(self):

        active_world_cells = []#added
        if self.custom_bounds is not None:
            position = [random.randint(self.custom_bounds[0], min(self.the_world.num_columns-self.dimensions[0], self.custom_bounds[1])),
                        random.randint(self.custom_bounds[2], min(self.the_world.num_rows-self.dimensions[1], self.custom_bounds[3]))]
        else:
            position = [random.randint(0, self.the_world.num_columns-self.dimensions[0]),
                        random.randint(0, self.the_world.num_rows-self.dimensions[1])]

        # checks to make sure theres not already a shape in that spot
        active_world_cells = self.get_active_world_cells(position)

        return position, active_world_cells  # added


    def get_active_world_cells(self, position):
        active_world_cell_list = []
        for i in range(len(self.active_cell_list)):
            new_cell = (self.active_cell_list[i][0] + position[0],
                        self.active_cell_list[i][1] + position[1])
            active_world_cell_list.append(new_cell)
        return active_world_cell_list

    def get_dimensions(self):
        if self.size == 1:
            height = 1
            width = 1
        else:
            min_val = [self.active_cell_list[0][0], self.active_cell_list[0][1]]
            max_val = [self.active_cell_list[0][0], self.active_cell_list[0][1]]
            for i in range(len(self.active_cell_list)):
                if self.active_cell_list[i][0] < min_val[0]:
                    min_val[0] = self.active_cell_list[i][0]
                if self.active_cell_list[i][1] < min_val[1]:
                    min_val[1] = self.active_cell_list[i][1]
                if self.active_cell_list[i][0] > max_val[0]:
                    max_val[0] = self.active_cell_list[i][0]
                if self.active_cell_list[i][1] > max_val[1]:
                    max_val[1] = self.active_cell_list[i][1]
            width = max_val[0] - min_val[0] + 1
            height = max_val[1] - min_val[1] + 1
        self.dimensions = (width, height)

    def take_turn(self):

        done = False
        try_counter = 0
        while not done:
            if try_counter > 100:
                print("Failed to flip or rotate after 100 tries")
                break

            action_choice = np.random.choice(self.action_list, 1, p=self.action_probs)
            if action_choice == 'rest':
                done = self.rest()

            elif action_choice == 'move':
                direction = random.choice([(0, 1), (0, -1), (-1, 0), (1, 0)])
                done = self.move(direction)

            elif action_choice == 'rotate':
                direction = random.choice([0, 1])
                done = self.rotate(direction)

            elif action_choice == 'flip':
                direction = random.choice([0, 1])
                done = self.flip(direction)
            try_counter += 1

    def move(self, direction):
        new_position = [self.position[0] + direction[0], self.position[1] + direction[1]]
        new_active_world_cell_list = self.get_active_world_cells(new_position)
        legal_position = self.the_world.check_legal_position(new_active_world_cell_list, self.id_number)
        if legal_position:
            self.commit_action("move", self.current_variant, new_position, new_active_world_cell_list)
        return legal_position

    def rotate(self, direction):
        rotation_variant = self.rotation_dict[self.current_variant]
        new_variant = rotation_variant[direction]
        new_active_cell_list = self.active_cell_dict[new_variant]
        new_active_world_cell_list = []
        for i in range(self.size):
            new_cell = (new_active_cell_list[i][0] + self.position[0], new_active_cell_list[i][1] + self.position[1])
            new_active_world_cell_list.append(new_cell)
        legal_position = self.the_world.check_legal_position(new_active_world_cell_list, self.id_number)
        if legal_position:
            self.commit_action("rotate", new_variant, self.position, new_active_world_cell_list)
        return legal_position

    def flip(self, direction):
        new_variant = self.flip_dict[self.current_variant][direction]
        new_active_cell_list = self.active_cell_dict[new_variant]
        new_active_world_cell_list = []
        for i in range(self.size):
            new_cell = (new_active_cell_list[i][0] + self.position[0], new_active_cell_list[i][1] + self.position[1])
            new_active_world_cell_list.append(new_cell)
        legal_position = self.the_world.check_legal_position(new_active_world_cell_list, self.id_number)
        if legal_position:
            self.commit_action("flip", new_variant, self.position, new_active_world_cell_list)
        return legal_position

    def rest(self):
        self.commit_action("rest", self.current_variant, self.position, self.active_world_cell_list)
        return True

    def commit_action(self, action_choice, current_variant, position, new_active_world_cell_list):
        for cell in self.active_world_cell_list:
            self.the_world.occupied_cell_dict.pop(cell)
        for cell in new_active_world_cell_list:
            self.the_world.occupied_cell_dict[cell] = self.id_number
        self.action_choice = action_choice
        self.current_variant = current_variant
        self.position = position
        self.active_world_cell_list = new_active_world_cell_list


#################################################################
class Monomino(Shape):

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "monomino"
        self.size = 1

        self.num_variants = 1
        self.variant_list = [0]
        self.active_cell_dict = {0: [(0, 0)]}
        self.flip_dict = {0: (0, 0)}
        self.rotation_dict = {0: (0, 0)}


#################################################################
class Domino(Shape):

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "domino"
        self.size = 2

        self.num_variants = 2
        self.variant_list = [0, 1]
        self.active_cell_dict = {0: [(0, 0), (0, 1)],
                                 1: [(0, 0), (1, 0)]}
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}
        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


#################################################################
class Tromino1(Shape):

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tromino1"  # l
        self.size = 3

        self.num_variants = 2
        self.variant_list = [0, 1]
        self.active_cell_dict = {0: [(0, 0), (1, 0), (2, 0)],
                                 1: [(0, 0), (0, 1), (0, 2)]}
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}

        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


#################################################################
class Tromino2(Shape):

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tromino2" # L
        self.size = 3

        self.num_variants = 4
        self.variant_list = [0, 1, 2, 3]
        self.active_cell_dict = {0: [(0, 0), (0, 1), (1, 0)],  # missing top right
                                 1: [(0, 0), (0, 1), (1, 1)],  # missing bottom right
                                 2: [(0, 1), (1, 0), (1, 1)],  # missing bottom left
                                 3: [(0, 0), (1, 0), (1, 1)],  # missing top left
                                 }

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (3, 1),
                          1: (2, 0),
                          2: (1, 3),
                          3: (0, 2)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 3),
                              1: (2, 0),
                              2: (3, 1),
                              3: (0, 2)}


#################################################################
class Tetromino1(Shape):
    # square

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tetromino1"
        self.size = 4

        self.num_variants = 1
        self.variant_list = [0]
        self.active_cell_dict = {0: [(0, 0), (0, 1), (1, 0), (1, 1)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (0, 0)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (0, 0)}


#################################################################
class Tetromino2(Shape):
    # line

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tetromino2"
        self.size = 4

        self.num_variants = 2
        self.variant_list = [0, 1]
        self.active_cell_dict = {0: [(0, 0), (0, 1), (0, 2), (0, 3)],
                                 1: [(0, 0), (1, 0), (2, 0), (3, 0)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


#################################################################
class Tetromino3(Shape):
    # squat T

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tetromino3"
        self.size = 4

        self.num_variants = 4
        self.variant_list = [0, 1, 2, 3]
        self.active_cell_dict = {0: [(0, 0), (1, 0), (2, 0), (1, 1)],
                                 1: [(0, 0), (0, 1), (0, 2), (1, 1)],
                                 2: [(0, 1), (1, 1), (2, 1), (1, 0)],
                                 3: [(1, 0), (1, 1), (1, 2), (0, 1)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (0, 2),
                          1: (3, 1),
                          2: (2, 0),
                          3: (1, 3)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 3),
                              1: (2, 2),
                              2: (3, 1),
                              3: (0, 0)}


#################################################################
class Tetromino4(Shape):
    # L

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = 'tetromino4'
        self.size = 4

        self.num_variants = 8
        self.variant_list = [0, 1, 2, 3, 4, 5, 6, 7]
        self.active_cell_dict = {0: [(0, 0), (0, 1), (1, 1), (2, 1)],
                                 1: [(0, 2), (1, 2), (1, 1), (1, 0)],
                                 2: [(0, 0), (1, 0), (2, 0), (2, 1)],
                                 3: [(0, 0), (0, 1), (0, 2), (1, 0)],

                                 4: [(0, 0), (1, 0), (2, 0), (0, 1)],
                                 5: [(0, 0), (0, 1), (0, 2), (1, 2)],
                                 6: [(0, 1), (1, 1), (2, 1), (2, 0)],
                                 7: [(0, 0), (1, 0), (1, 1), (1, 2)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (6, 4),
                          1: (5, 7),
                          2: (4, 6),
                          3: (7, 5),
                          4: (2, 0),
                          5: (1, 3),
                          6: (0, 2),
                          7: (3, 1)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 3),
                              1: (2, 0),
                              2: (3, 1),
                              3: (0, 2),
                              4: (5, 7),
                              5: (6, 4),
                              6: (7, 5),
                              7: (4, 6)
                              }


#################################################################
class Tetromino5(Shape):
    # z

    def __init__(self, the_world):
        super().__init__(the_world)

        self.name = "tetromino5"
        self.size = 4

        self.num_variants = 4
        self.variant_list = [0, 1, 2, 3]
        self.active_cell_dict = {0: [(0, 0), (0, 1), (1, 1), (1, 2)],
                                 1: [(0, 1), (1, 0), (1, 1), (2, 0)],
                                 2: [(0, 1), (0, 2), (1, 0), (1, 1)],
                                 3: [(0, 0), (1, 0), (1, 1), (2, 1)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (2, 2),
                          1: (3, 3),
                          2: (0, 0),
                          3: (1, 1)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 1),
                              1: (0, 0),
                              2: (3, 3),
                              3: (2, 2)}
