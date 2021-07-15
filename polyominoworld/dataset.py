import random
import numpy as np
from typing import List

from polyominoworld.helpers import Event, Sequence


class DataSet:
    """generates vectorized training samples given event sequences that take place in the world"""

    def __init__(self,
                 sequences: List[Sequence],
                 shuffle_sequences: bool,
                 shuffle_events: bool,
                 seed: int,
                 name: str,
                 ) -> None:

        np.random.seed(seed)

        self.sequences = sequences
        self.shuffle_sequences = shuffle_sequences
        self.shuffle_events = shuffle_events
        self.name = name

        print(f'Initialized {self.name} dataset with {len(self.sequences):,} sequences')

    def get_events(self,
                   ) -> List[Event]:
        """
        return list of events, for training, testing, visualisation, etc.
        """
        res = []

        if self.shuffle_sequences:
            random.shuffle(self.sequences)

        # for each sequence
        for sequence in self.sequences:

            if self.shuffle_events:
                random.shuffle(sequence.events)

            # for each event
            for event in sequence.events:
                res.append(event)

        return res

    def __len__(self):
        return len(list(self.get_events()))
