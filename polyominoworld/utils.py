import itertools
from typing import List, Tuple

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


def calc_terms1_and_terms2(q: mp.Queue,
                           data: DataSet,
                           h_x: np.array,
                           scale_weights: float,
                           rgb_id: int,
                           score_avg_max: mp.Value,
                           ):
    """
    a consumer that reads input from a queue and saves best results to shared memory.

    multiple consumers ca be used to find combinations of input weight detectors that result in highest score.
    the score is related to overlap between shapes of hidden states corresponding to examples of a shape

    """

    best_h_ids_and_best_score = [None, 0.0]

    while True:

        h_ids = q.get()

        if h_ids is None:
            break

        # get set of detectors for single color channel
        detectors = []
        for h_id, hi in enumerate(h_x):
            # get weights to one color channel only
            hi_single_channel = hi.reshape((3, configs.World.max_x, configs.World.max_y))[rgb_id, :, :]
            # scale so that modes of hi are aligned with integers in base 1
            hi_single_channel_scaled = hi_single_channel * scale_weights
            # make discrete
            hi_single_channel_discrete = np.rint(hi_single_channel_scaled)
            if h_id in h_ids:
                detectors.append(hi_single_channel_discrete.flatten())
        detector_mat = np.array(detectors)

        # compute states produced by dot product of input and detectors
        shape2states = {shape: [] for shape in configs.World.master_shapes}
        shape2states_other = {shape: [] for shape in configs.World.master_shapes}
        for event in data.get_events():
            # get input from single color channel
            x = event.world_vector.as_3d()[rgb_id].flatten()
            # compute discrete state
            state = tuple(detector_mat @ x)
            # collect
            shape2states[event.shape].append(state)
            for shape_other in configs.World.master_shapes:
                if shape_other != event.shape:
                    shape2states_other[shape_other].append(state)

        # analyze states for invariance to rotation and location, for each shape separately
        scores = []
        for shape, states in sorted(shape2states.items()):
            num_states_unique_to_shape = 0
            for state_i in states:
                if state_i not in shape2states_other[shape]:
                    num_states_unique_to_shape += 1
            score = num_states_unique_to_shape / len(states)
            # collect
            scores.append(score)

        # compute average score
        score_avg = np.mean(scores)
        is_best = ' ' if score_avg < score_avg_max.value else 'T'
        scores_formatted = ' '.join([f'{s:.2f}' for s in scores])
        if score_avg > score_avg_max.value:
            print(f'avg={score_avg :.4f} best={is_best} rgb_id={rgb_id} {scores_formatted} h_ids={h_ids} ')

        # update shared memory
        if score_avg > score_avg_max.value:
            score_avg_max.value = score_avg
            best_h_ids_and_best_score = [h_ids, score_avg_max.value]

    print(f'best score={best_h_ids_and_best_score[1]:.4f} for consumer with best h_ids={best_h_ids_and_best_score[0]}')
