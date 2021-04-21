import itertools
from typing import List, Tuple, Optional
import numpy as np
import multiprocessing as mp

from polyominoworld import configs
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network


def get_leftout_positions(leftout_half: str,
                          ) -> List[Tuple[int, int]]:
    """get positions in world that are in leftout half of the world"""

    all_positions = [(x, y) for x, y in
                     itertools.product(range(configs.World.max_x), range(configs.World.max_y))]

    if leftout_half == 'lower':
        return [(x, y) for x, y in all_positions
                if y < configs.World.max_y / 2]
    elif leftout_half == 'upper':
        return [(x, y) for x, y in all_positions
                if y >= configs.World.max_y / 2]
    elif leftout_half == '':
        return []  # nothing is leftout
    else:
        raise AttributeError('Invalid arg to leftout_half')


def evaluate_detector_combo(q: mp.Queue,
                            data: DataSet,
                            net: Network,
                            feature_type: str,
                            score_avg_max: mp.Value,
                            lock: mp.Lock,
                            ):
    """
    a consumer that reads input from a queue and saves best results to shared memory.

    multiple consumers can be used to find combinations of detectors that result in highest score.

    """

    from polyominoworld.evaluate import evaluate_linear_readout

    while True:

        h_ids = q.get()

        if h_ids is None:
            break

        score = evaluate_linear_readout(data, net, feature_type, h_ids)

        # TODO test lock
        lock.acquire()
        # report + update shared memory
        if score > score_avg_max.value:
            print(f'score={score :.4f} | h_ids={h_ids} ')
        if score > score_avg_max.value:
            score_avg_max.value = score
        lock.release()
