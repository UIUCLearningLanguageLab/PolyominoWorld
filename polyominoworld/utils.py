import itertools
from typing import List, Tuple, Optional

import numpy as np
import multiprocessing as mp

from polyominoworld import configs
from polyominoworld.dataset import DataSet


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
                            h_x: np.array,
                            scale_weights: float,
                            score_avg_max: mp.Value,
                            rgb_id: Optional[int] = None,
                            across_shape: bool = True,
                            ):
    """
    a consumer that reads input from a queue and saves best results to shared memory.

    multiple consumers can be used to find combinations of detectors that result in highest score.
    the score is related to overlap between shapes of hidden states corresponding to examples of a shape

    """
    if rgb_id is not None:
        name_of_color_channel = {0: 'red', 1: 'green', 2: 'blue'}[rgb_id]
    else:
        name_of_color_channel = None

    while True:

        h_ids = q.get()

        if h_ids is None:
            break

        # get set of detectors for one or all color channel
        detectors = []
        for h_id, hi in enumerate(h_x):
            # get weights to one color or all channel
            if rgb_id is not None:
                hi_reshaped = hi.reshape((3, configs.World.max_x, configs.World.max_y))[rgb_id, :, :]
            else:
                hi_reshaped = hi
            # scale so that modes of hi are aligned with integers in base 1
            hi_scaled = hi_reshaped * scale_weights
            # make discrete
            hi_discrete = np.rint(hi_scaled)
            # collect detectors only if they are supposed to be collected
            if h_id in h_ids:
                detectors.append(hi_discrete.flatten())
        detector_mat = np.array(detectors)

        # compute states produced by dot product of input and detectors
        shape2states = {shape: [] for shape in configs.World.master_shapes}
        shape2states_other = {shape: [] for shape in configs.World.master_shapes}
        for event in data.get_events():
            # filter by color channel
            if rgb_id is not None and event.color != name_of_color_channel:
                continue
            # get input from single color channel
            if rgb_id is not None:
                x = event.world_vector.as_3d()[rgb_id].flatten()
            else:
                x = event.world_vector.vector.numpy()
            # compute discrete state
            state = tuple(detector_mat @ x)
            # collect states
            shape2states[event.shape].append(state)
            for shape_other in configs.World.master_shapes:
                if shape_other != event.shape:
                    shape2states_other[shape_other].append(state)

        # compute score: how often are states for one shape shared by other shapes, on average?
        scores = []
        no_states = False
        for shape, states in sorted(shape2states.items()):
            if not states:  # this may happen if no data for a particular color is in data
                no_states = True
                break

            if across_shape:
                num_times_confused = 0
                for state in states:
                    num_times_confused += shape2states_other[shape].count(state)
                score = len(states) / (num_times_confused + len(states))

            else:
                num_times_confused = 0
                for state in states:
                    num_times_confused += states.count(state)
                score = num_times_confused / (len(states) * len(states))

            # collect
            scores.append(score)

        if no_states:
            continue

        # compute average score
        score_avg = np.mean(scores)
        scores_formatted = ' '.join([f'{s:.2f}' for s in scores])
        if score_avg > score_avg_max.value:
            print(f'avg={score_avg :.4f} rgb_id={rgb_id} | {scores_formatted} | h_ids={h_ids} ')

        # update shared memory
        if score_avg > score_avg_max.value:
            score_avg_max.value = score_avg
