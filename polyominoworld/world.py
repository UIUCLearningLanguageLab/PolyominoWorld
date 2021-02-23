from typing import Tuple, List, Dict
import numpy as np

from polyominoworld import configs
from polyominoworld.params import Params
from polyominoworld import shapes
from polyominoworld.helpers import Sequence, Event, ShapeState, FeatureVector, WorldVector, WorldCell


RGB = Tuple[float, float, float]


class World:
    """
    a grid-based world in which sequences of events occur.
    events occur because to actions performed by shapes, e.g. flip, move, rotate
    """

    def __init__(self,
                 params: Params,
                 ) -> None:

        self.params = params

        self.actions = [a for a, p in self.params.actions_and_probabilities]
        self.action_probabilities = [p for a, p in self.params.actions_and_probabilities]

        # init/reset world by starting with no active cell
        self.active_cell2color: Dict[WorldCell, RGB] = {}

    def generate_sequences(self) -> List[Sequence]:
        """generate sequences of events, each with one shape"""

        res = []

        # for each possible color
        for color in configs.World.color2rgb:  # TODO specify which colors to use in params.py

            if color == self.params.bg_color:
                continue

            # for each user-requested shape, variant combination
            for shape_name, variants in self.params.shapes_and_variants:
                for variant in variants:

                    # for each possible location
                    for pos_x in range(configs.World.max_y):
                        for pos_y in range(configs.World.max_x):

                            # make shape
                            position = (pos_x, pos_y)
                            shape = self._make_shape(color, position, shape_name, variant)

                            if not self._is_shape_legal(shape):
                                continue

                            # update occupied cells
                            for cell in self._calc_active_world_cells(shape):
                                self.active_cell2color[cell] = shape.color

                            # make sequence of events
                            sequence = self._make_sequence(shape)
                            res.append(sequence)

        return res

    @staticmethod
    def _make_shape(color: str,
                    position: Tuple[int, int],
                    name: str,
                    variant: int,
                    ) -> shapes.Shape:
        if name == 'monomino':
            constructor = shapes.Monomino
        elif name == 'domino':
            constructor = shapes.Domino
        elif name == 'tromino1':
            constructor = shapes.Tromino1
        elif name == 'tromino2':
            constructor = shapes.Tromino2
        elif name == 'tetromino1':
            constructor = shapes.Tetromino1
        elif name == 'tetromino2':
            constructor = shapes.Tetromino2
        elif name == 'tetromino3':
            constructor = shapes.Tetromino3
        elif name == 'tetromino4':
            constructor = shapes.Tetromino4
        elif name == 'tetromino5':
            constructor = shapes.Tetromino5
        else:
            raise AttributeError('Invalid arg to shape name')

        return constructor(color, variant, position)

    def _make_sequence(self, shape):
        """a sequence of events that involve a single shape"""

        events = []
        for event_id in range(self.params.num_events_per_sequence):

            # find and perform legal action
            shape_state = self.find_legal_shape_state(shape)
            shape.update_state(shape_state)

            # calculate new active world cells + update occupied cells
            new_active_world_cells = self._calc_active_world_cells(shape)
            self._update_active_cells(new_active_world_cells, shape.color)
            assert len(self.active_cell2color) == shape.size

            # collect event that resulted from action
            events.append(
                Event(
                    shape=shape.name,
                    size=shape.size,
                    color=shape.color,
                    variant=shape.variant,
                    pos_x=shape.pos_x,
                    pos_y=shape.pos_y,
                    action=shape.action,
                    world_vector=WorldVector.from_world(self),
                    hidden_vector=NotImplemented,  # TODO
                    feature_vector=FeatureVector.from_shape(shape),
                ))

        res = Sequence(events)
        return res

    @staticmethod
    def _calc_active_world_cells(shape: shapes.Shape,
                                 ) -> List[WorldCell]:
        """
        compute the cells in the world that this shape would occupy
        """
        res = []
        for cell in shape.active_cells:
            res.append(WorldCell(x=cell.x + shape.pos_x,
                                 y=cell.y + shape.pos_y))

        assert len(res) == shape.size
        return res

    def _is_shape_legal(self,
                        shape: shapes.Shape,
                        ) -> bool:
        """
        note: because only one shape can occupy world at any time,
         the only requirement for legality is being within bounds of the world
        """

        for cell in self._calc_active_world_cells(shape):

            # not legal if out of bounds
            if (cell.x < 0) or \
                    (cell.y < 0) or \
                    (cell.x > configs.World.max_x - 1) or \
                    (cell.y > configs.World.max_y - 1):
                return False

        return True

    def find_legal_shape_state(self,
                               shape: shapes.Shape,
                               ) -> ShapeState:

        """find shape state that results in legal position"""

        try_counter = 0
        while True:
            if try_counter > configs.Try.max:
                raise RuntimeError(f"Did not find legal position after {configs.Try.max} tries")

            # random action
            action = np.random.choice(self.actions, 1, p=self.action_probabilities).item()
            state: ShapeState = shape.get_new_state(action)

            # update world and cell if resultant position is legal
            if self._is_shape_legal(shape):
                return state

            try_counter += 1

    def _update_active_cells(self,
                             world_cells: List[WorldCell],
                             color: RGB,
                             ) -> None:

        self.active_cell2color.clear()

        for cell in world_cells:
            self.active_cell2color[cell] = color
