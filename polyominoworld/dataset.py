import random
import numpy as np
from typing import List

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

        print(f'Initialized {self.name} dataset with {len(self.sequences):,} sequences')

    def get_events(self,
                   ) -> List[Event]:
        """
        return list of events, for training, testing, visualisation, etc.
        """
        res = []

        if self.params.shuffle_sequences:
            random.shuffle(self.sequences)

        # for each sequence
        for sequence in self.sequences:

            if self.params.shuffle_events:
                random.shuffle(sequence.events)

            # for each event
            for event in sequence.events:
                res.append(event)

        return res

    def __len__(self):
        return len(list(self.get_events()))
