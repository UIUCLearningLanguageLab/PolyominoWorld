from dataclasses import dataclass, field
from typing import Tuple, List, Dict
import numpy as np
import torch

from polyominoworld import configs


@dataclass(frozen=True)
class WorldCell:
    x: int = field()
    y: int = field()


@dataclass(frozen=True)
class ShapeCell:
    x: int = field()
    y: int = field()


color2channels4 = {c: np.hstack((np.mean(rgb), np.array(rgb))) for c, rgb in configs.World.color2rgb.items()}
color2channels3 = {c: np.array(rgb) for c, rgb in configs.World.color2rgb.items()}


@dataclass()
class WorldVector:
    """
    raw world data encoded in a single vector
    """

    active_cell2color: Dict[WorldCell, str] = field()
    bg_color: str = field()
    add_grayscale: bool = field()
    shift_input: int = field()  # todo experimental

    def _make_bg_color_vector(self,
                              add_grayscale: bool,
                              ) -> np.array:
        """
        create vector of size 3 or 4, corresponding to channel values of background color (rgb or rgb + grayscale).

        note: background color will be different every time this function is called if bg_color='random'
        """
        if self.bg_color == 'random':
            if add_grayscale:
                res = np.random.uniform(0, 1, size=4)
            else:
                res = np.random.uniform(0, 1, size=3)
        else:
            if self.bg_color not in configs.World.color2rgb:
                raise RuntimeError(f"Color {self.bg_color} not recognized")

            if add_grayscale:
                res = color2channels4[self.bg_color]
            else:
                res = color2channels3[self.bg_color]

        return res

    @property
    def vector(self) -> torch.tensor:
        """
        return a vector that represents the world, by
        concatenating the channel vector (of size 3 if not grayscale, else 4) for each cell in world.
        the rgb vector corresponds to either a background color or the color of a shape.
        """

        channel_vectors = []
        for pos_x in range(configs.World.max_x):
            for pos_y in range(configs.World.max_y):
                cell = WorldCell(pos_x, pos_y)

                # shape at cell
                if cell in self.active_cell2color:
                    color: str = self.active_cell2color[cell]
                    if self.add_grayscale:
                        channels_vector = color2channels4[color]
                    else:
                        channels_vector = color2channels3[color]

                # background
                else:
                    channels_vector = self._make_bg_color_vector(self.add_grayscale)

                # todo experimental
                if self.shift_input > 0:
                    channels_vector = channels_vector.copy()  # otherwise value in color2channel is changed
                    channels_vector += self.shift_input

                channel_vectors.append(channels_vector)

        # concatenate channel_vectors
        concatenation = np.hstack(channel_vectors).astype(np.float32)
        res = torch.from_numpy(concatenation)

        if configs.Device.gpu:
            res = res.cuda()

        return res

    def as_3d(self) -> np.ndarray:
        """
        return world vector as 3d array, where first index, of size 3, corresponds to RGB values.
        the returned array has shape (3, max_x, max_y)

        note: used for visualisation, not training. this array always has 3 rgb channels, and never adds grayscale.

        note: column indices should always be considered as  x coordinates,
        and row indices should always be considered as y coordinates.
        """

        res = np.zeros((3, configs.World.max_x, configs.World.max_y))

        for pos_x in range(configs.World.max_x):
            for pos_y in range(configs.World.max_y):
                cell = WorldCell(pos_x, pos_y)

                # color of shape at cell
                if cell in self.active_cell2color:
                    color = self.active_cell2color[cell]
                    res[:, pos_y, pos_x] = color2channels3[color]  # always use rgb only

                # color of background
                else:
                    res[:, pos_y, pos_x] = self._make_bg_color_vector(add_grayscale=False)

        return res

    @classmethod
    def from_world(cls,
                   world,
                   ):
        """
        warning: it is extremely important to copy active_cell2color,
         otherwise it will be linked to the world, and updated whenever the world is updated,
        """
        return cls(world.active_cell2color.copy(),
                   world.params.bg_color,
                   world.params.add_grayscale,
                   world.params.shift_input,
                   )


@dataclass(frozen=True)
class FeatureLabel:

    type_name: str = field()
    value_name: str = field()

    def __str__(self):
        return f'{self.type_name}-{self.value_name}'


@dataclass(frozen=True)
class FeatureVector:
    """
    features, e.g. shape encoded in a vector.
    note: each individual vector is a one-hot vector
    """

    shape_vector: np.array = field()
    size_vector: np.array = field()
    color_vector: np.array = field()
    action_vector: np.array = field()

    @classmethod
    def calc_size(cls) -> int:
        res = 0
        for values in configs.World.feature_type2values.values():
            res += len(values)
        return res

    @classmethod  # this function should be callable without instantiating the class
    def get_feature_labels(cls) -> List[FeatureLabel]:
        """return names of features, one for each output in feature vector.

        note: each feature is named by concatenating:
         - the name of the feature type (e.g. shape)
         - the name of the feature value (e.g. red)
         """

        res = []
        for feature_type, values in configs.World.feature_type2values.items():
            for feature_value in values:
                label = FeatureLabel(feature_type, feature_value)
                res.append(label)
        return res

    @property
    def feature_type2ids(self) -> Dict[str, List[int]]:
        """
        dict that maps feature_type to indices into feature vector where features of feature_type are located
        """
        res = {}
        start = 0
        for feature_type, values in configs.World.feature_type2values.items():
            stop = start + len(values)
            res[feature_type] = list(range(start, stop))
            start += len(values)
        return res

    @property
    def vector(self) -> torch.tensor:
        res = np.hstack(
            # order matters: shape, size, color, action
            (self.shape_vector, self.size_vector, self.color_vector, self.action_vector)
        ).astype(np.float32)

        res = torch.from_numpy(res)

        if configs.Device.gpu:
            res = res.cuda()

        return res

    @classmethod
    def from_shape(cls,
                   shape,
                   ):
        # create one-hot vectors
        shape_vector = np.eye(len(configs.World.feature_type2values['shape']),
                              dtype=np.int32)[configs.World.feature_type2values['shape'].index(shape.name)]
        size_vector = np.eye(len(configs.World.feature_type2values['size']),
                             dtype=np.int32)[configs.World.feature_type2values['size'].index(shape.size)]
        color_vector = np.eye(len(configs.World.feature_type2values['color']),
                              dtype=np.int32)[configs.World.feature_type2values['color'].index(shape.color)]
        action_vector = np.eye(len(configs.World.feature_type2values['action']),
                               dtype=np.int32)[configs.World.feature_type2values['action'].index(shape.action)]

        return cls(shape_vector, size_vector, color_vector, action_vector)


@dataclass(frozen=True)
class ShapeState:
    """state of a shape"""
    action: str = field()
    variant: int = field()
    position: Tuple[int, int] = field()


@dataclass(frozen=True)
class Event:
    """a time-slice of a sequence"""
    shape: str = field()  # shape name
    size: int = field()
    color: str = field()
    variant: int = field()
    pos_x: int = field()
    pos_y: int = field()
    action: str = field()
    world_vector: WorldVector = field()
    hidden_vector: NotImplemented = field()
    feature_vector: FeatureVector = field()

    def get_label(self,
                  feature_type: str,
                  ) -> str:
        return self.__dict__[feature_type]

    def get_x(self,
              x_type: str,
              ) -> torch.tensor:
        if x_type == 'hidden':
            return self.hidden_vector.vector
        elif x_type == 'world':
            return self.world_vector.vector
        else:
            raise AttributeError(f'Invalid arg to x_type')

    def get_y(self,
              y_type: str):
        if y_type == 'features':
            return self.feature_vector.vector
        elif y_type == 'world':
            return self.world_vector.vector
        else:
            raise AttributeError(f'Invalid arg to y_type')


@dataclass(frozen=True)
class Sequence:
    """a sequence of events"""
    events: List[Event] = field()

    def __iter__(self):
        return self.events
