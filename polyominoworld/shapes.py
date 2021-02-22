import random
from typing import Tuple, List

from polyominoworld import configs
from polyominoworld.helpers import ShapeState, ShapeCell


RGB = Tuple[float, float, float]


class Shape:

    def __init__(self,
                 color: RGB,
                 variant: int,
                 position: Tuple[int, int],
                 ) -> None:

        self.color = color
        self.variant = variant
        self.position = position
        self.pos_x, self.pos_y = self.position
        self.action = 'rest'

        # attributes set by sub-class
        self.name = NotImplemented
        self.size = NotImplemented
        self.variant2active_cells = NotImplemented
        self.flip_dict = NotImplemented
        self.rotation_dict = NotImplemented

    @property
    def active_cells(self) -> List[ShapeCell]:
        return [ShapeCell(*cell) for cell in self.variant2active_cells[self.variant]]

    @property
    def dimensions(self):
        """
        calculate the the shape's width and height given its current variant
        """
        if self.size == 1:
            height = 1
            width = 1

        else:
            min_x, min_y = self.active_cells[0].x, self.active_cells[0].y  # todo this seems like a hack
            max_x, max_y = self.active_cells[0].x, self.active_cells[0].y

            for active_cell in self.active_cells:
                if active_cell.x < min_x:
                    min_x = active_cell.x
                if active_cell.y < min_y:
                    min_y = active_cell.y
                if active_cell.x > max_x:
                    max_x = active_cell.x
                if active_cell.y > max_y:
                    max_y = active_cell.y
            width = max_x - min_x + 1
            height = max_y - min_y + 1

        return width, height

    @property
    def variants(self) -> List[int]:
        return [k for k in self.variant2active_cells]

    @property
    def num_variants(self) -> int:
        return len(self.variants)

    def get_new_state(self,
                      action: str,
                      ):

        if action == 'rest':
            return self.get_new_state_after_rest()

        directions = configs.World.action2directions[action]
        direction = random.choice(directions)

        if action == 'move':
            return self.get_new_state_after_move(direction)
        elif action == 'rotate':
            return self.get_new_state_after_rotate(direction)
        elif action == 'flip':
            return self.get_new_state_after_flip(direction)
        else:
            raise AttributeError('Invalid action')

    def get_new_state_after_move(self, direction) -> ShapeState:
        new_position = (self.pos_x + direction[0], self.pos_y + direction[1])
        return ShapeState(action="move",
                          variant=self.variant,
                          position=new_position,
                          )

    def get_new_state_after_rotate(self, direction) -> ShapeState:
        rotation_variant = self.rotation_dict[self.variant]
        new_variant = rotation_variant[direction]
        return ShapeState(action="rotate",
                          variant=new_variant,
                          position=self.position,
                          )

    def get_new_state_after_flip(self, direction) -> ShapeState:
        new_variant = self.flip_dict[self.variant][direction]
        return ShapeState(action="flip",
                          variant=new_variant,
                          position=self.position,
                          )

    def get_new_state_after_rest(self) -> ShapeState:
        return ShapeState(action="rest",
                          variant=self.variant,
                          position=self.position,
                          )

    def update_state(self,
                     state: ShapeState,
                     ) -> None:
        """note: active_cells is automatically updated when variant is updated"""
        self.action = state.action
        self.variant = state.variant
        self.position = state.position


class Monomino(Shape):

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "monomino"
        self.size = 1

        self.variant2active_cells = {0: [(0, 0)]}
        self.flip_dict = {0: (0, 0)}
        self.rotation_dict = {0: (0, 0)}


class Domino(Shape):

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "domino"
        self.size = 2

        self.variant2active_cells = {0: [(0, 0), (0, 1)],
                                     1: [(0, 0), (1, 0)]}
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}
        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


class Tromino1(Shape):

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tromino1"  # l
        self.size = 3

        self.variant2active_cells = {0: [(0, 0), (1, 0), (2, 0)],
                                     1: [(0, 0), (0, 1), (0, 2)]}
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}

        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


class Tromino2(Shape):

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tromino2"  # L
        self.size = 3

        self.variant2active_cells = {0: [(0, 0), (0, 1), (1, 0)],  # missing top right
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


class Tetromino1(Shape):
    # square

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tetromino1"
        self.size = 4

        self.variant2active_cells = {0: [(0, 0), (0, 1), (1, 0), (1, 1)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (0, 0)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (0, 0)}


class Tetromino2(Shape):
    # line

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tetromino2"
        self.size = 4

        self.variant2active_cells = {0: [(0, 0), (0, 1), (0, 2), (0, 3)],
                                     1: [(0, 0), (1, 0), (2, 0), (3, 0)]}

        # (flip sideways, flip top-ways)
        self.flip_dict = {0: (0, 0),
                          1: (1, 1)}

        # (rotate clockwise, rotate counterclockwise)
        self.rotation_dict = {0: (1, 1),
                              1: (0, 0)}


class Tetromino3(Shape):
    # squat T

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tetromino3"
        self.size = 4

        self.variant2active_cells = {0: [(0, 0), (1, 0), (2, 0), (1, 1)],
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


class Tetromino4(Shape):
    # L

    def __init__(self, *args):
        super().__init__(*args)

        self.name = 'tetromino4'
        self.size = 4

        self.variant2active_cells = {0: [(0, 0), (0, 1), (1, 1), (2, 1)],
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


class Tetromino5(Shape):
    # z

    def __init__(self, *args):
        super().__init__(*args)

        self.name = "tetromino5"
        self.size = 4

        self.variant2active_cells = {0: [(0, 0), (0, 1), (1, 1), (1, 2)],
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
