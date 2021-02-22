import random
import numpy as np
from typing import Generator, Tuple, List

from polyominoworld import configs
from polyominoworld.params import Params
from polyominoworld.helpers import Event, Sequence


class DataSet:
    """generates vectorized training samples given event sequences that take place in the world"""

    def __init__(self,
                 sequences: List[Sequence],
                 params: Params,
                 name: str,
                 ) -> None:

        if params.y_type not in configs.ArgCheck.y_type:
            raise AttributeError('Invalid arg to y_type')

        if params.x_type not in configs.ArgCheck.x_type:
            raise AttributeError('Invalid arg to x_type')

        np.random.seed(params.seed)

        self.sequences = sequences
        self.params = params
        self.name = name

        # colors
        self.master_color_labels = [color for color in configs.World.color2rgb if color != 'grey']
        self.color_rgb_matrix = self.make_color_rgb_matrix()

        print(f'Initialized {self.name} dataset with {len(self.sequences):,} sequences')

    def make_color_rgb_matrix(self):
        """create a num_colors x 3 matrix, with the RGB values for each color"""
        res = np.zeros([len(self.master_color_labels), 3], float)
        for n, color in enumerate(self.master_color_labels):
            rgb = configs.World.color2rgb[color]
            res[n] = rgb

        return res

    def generate_events(self,
                        ) -> Generator[Event, None, None]:
        """a generator that yields events, for training or testing.
         event.x is input vector.
         events.y is target output vector.
         """

        if self.params.shuffle_sequences:
            random.shuffle(self.sequences)

        # for each sequence
        for sequence in self.sequences:

            if self.params.shuffle_events:
                random.shuffle(sequence.events)

            # for each event
            for event in sequence.events:

                yield event

    def __len__(self):
        return len(list(self.generate_events()))
