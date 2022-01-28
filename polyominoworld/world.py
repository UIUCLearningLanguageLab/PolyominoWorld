from typing import Tuple, List, Dict
import numpy as np
import itertools

from polyominoworld import configs
from polyominoworld.params import Params
from polyominoworld import shapes
from polyominoworld.helpers import Sequence, Event, ShapeState, FeatureVector, WorldVector, WorldCell


class World:
    """
    a grid-based world in which sequences of events occur.
    events occur because to actions performed by shapes, e.g. flip, move, rotate


    the world is defined with respect to the origin (x=0, y=0), the top, left-most point
    """

    def __init__(self,
                 params: Params,
                 ) -> None:

        self.params = params

        # check params
        for color in self.params.fg_colors:
            if color not in configs.World.color2rgb:
                raise ValueError('Color must be in configs.World.color2rgb')

        self.actions = [a for a, p in self.params.actions_and_probabilities]
        self.action_probabilities = [p for a, p in self.params.actions_and_probabilities]

        # init/reset world by starting with no active cell
        self.active_cell2color: Dict[WorldCell, str] = {}

        self.master_positions = [(x, y) for x, y in
                                 itertools.product(range(configs.World.max_x), range(configs.World.max_y))]

    def generate_sequences(self,
                           leftout_colors: Tuple[str, ...],
                           leftout_shapes: Tuple[str, ...],
                           leftout_variants: str,
                           leftout_positions: List[Tuple[int, int]],
                           ) -> List[Sequence]:
        """generate sequences of events, each with one shape"""

        res = []

        # for each possible color
        for fg_color in self.params.fg_colors:

            if fg_color in leftout_colors:  # careful: do not leave out bg_color: test set will be empty
                continue

            if fg_color == self.params.bg_color:
                raise RuntimeError('Foreground color is background color')

            # for each user-requested shape+variant combination
            for shape_name, variants in self.params.shapes_and_variants:

                if shape_name in leftout_shapes:
                    continue

                # use only half of variants?
                if leftout_variants == 'half1':
                    variants = variants[:max(1, len(variants) // 2)]
                elif leftout_variants == 'half2':
                    variants = variants[len(variants) // 2:]
                elif leftout_variants == '':
                    pass
                else:
                    raise AttributeError(f'Invalid arg to leftout_variants: {leftout_variants}')

                for variant in variants:

                    # for each position in upper, or lower half, or both halves
                    for position in self.master_positions:

                        if position in leftout_positions:
                            continue

                        # make shape
                        shape = self._make_shape(fg_color, position, shape_name, variant)

                        if not self._is_shape_legal(shape, leftout_positions):
                            continue

                        # update occupied cells
                        for cell in self._calc_active_world_cells(shape):
                            self.active_cell2color[cell] = shape.color

                        # make sequence of events
                        sequence = self._make_sequence(shape,
                                                       leftout_positions,
                                                       self.params.leftout_feature_types)
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

    def _make_sequence(self,
                       shape,
                       leftout_positions: List[Tuple[int, int]],
                       leftout_feature_types: Tuple[str]
                       ):
        """a sequence of events that involve a single shape"""

        events = []
        for event_id in range(self.params.num_events_per_sequence):

            # find and perform legal action
            shape_state = self.find_legal_shape_state(shape, leftout_positions)
            shape.update_state(shape_state)

            # calculate new active world cells + update occupied cells
            new_active_world_cells = self._calc_active_world_cells(shape)
            self._update_active_cells(new_active_world_cells, shape.color)
            assert len(self.active_cell2color) == shape.size

            # collect event that resulted from action
            event = Event(
                shape=shape.name,
                size=shape.size,
                color=shape.color,
                variant=shape.variant,
                pos_x=shape.pos_x,
                pos_y=shape.pos_y,
                action=shape.action,
                world_vector=WorldVector.from_world(self, leftout_positions),
                hidden_vector=NotImplemented,  # TODO
                feature_vector=FeatureVector.from_shape(shape, leftout_feature_types),
            )
            events.append(event)

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
                        leftout_positions: List[Tuple[int, int]],
                        ) -> bool:
        """
        1. check if any (possibly out-of-bounds) cell occupied by the shape is out of bounds.
        2. check if any (possibly out-of-bounds) cell occupied by the shape crosses half boundary


        note: because only one shape can occupy world at any time,
         the only requirement for legality is being within bounds of the world,
         and not occupying any leftout position
        """

        for cell in self._calc_active_world_cells(shape):
            cell: WorldCell

            # not legal if out of bounds
            if (cell.x < 0) or \
                    (cell.y < 0) or \
                    (cell.x > configs.World.max_x - 1) or \
                    (cell.y > configs.World.max_y - 1):
                return False

            # check half boundary
            if cell in [WorldCell(x=pos[0], y=pos[1]) for pos in leftout_positions]:
                return False

        return True

    def find_legal_shape_state(self,
                               shape: shapes.Shape,
                               leftout_positions: List[Tuple[int, int]],
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
            if self._is_shape_legal(shape, leftout_positions):
                return state

            try_counter += 1

    def _update_active_cells(self,
                             world_cells: List[WorldCell],
                             color: str,
                             ) -> None:

        self.active_cell2color.clear()

        for cell in world_cells:
            self.active_cell2color[cell] = color
