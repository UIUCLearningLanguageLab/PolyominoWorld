import itertools
from typing import List, Tuple

import numpy as np
import multiprocessing as mp

from polyominoworld import configs
from polyominoworld.dataset import DataSet
from polyominoworld.figs import plot_hidden_weights_analysis, plot_state_analysis


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
                           largest_avg_res: mp.Value,
                           plot_weights: bool = False,
                           plot_states: bool = False,
                           ):
    """
    a consumer that reads input from a queue and saves best results to shared memory.

    multiple consumers ca be used to find combinations of input weight detectors that result in best "invariance"
    """

    while True:

        h_ids = q.get()

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
            # plot
            if plot_weights:
                x_tick_labels = [f'x{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[0])]
                y_tick_labels = [f'y{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[1])]
                plot_hidden_weights_analysis(hi_single_channel_scaled,
                                             title=f'pattern{h_id:0>3}\nregular={h_id in h_ids}',
                                             x_tick_labels=x_tick_labels,
                                             y_tick_labels=y_tick_labels)
        detector_mat = np.array(detectors)

        # compute states produced by dot product of input and detectors
        shape2states = {shape: [] for shape in configs.World.master_shapes}
        for event in data.get_events():
            # get input from single color channel
            x = event.world_vector.as_3d()[rgb_id].flatten()
            # compute discrete state
            state = detector_mat @ x
            shape2states[event.shape].append(state)

        # analyze states for invariance to rotation and location, for each shape separately
        terms1 = []
        terms2 = []
        for shape, states in sorted(shape2states.items()):
            states_mat = np.array(states)
            num_total_states = len(states_mat)
            unique_states = np.unique(states_mat, axis=0)
            # term 1: how many non-unique states does shape have?
            num_unique_states = len(unique_states)
            num_non_unique_states = num_total_states - num_unique_states
            term1 = num_non_unique_states / num_total_states
            # term 2: how many states are not shared by other shapes?
            num_shared = 0
            num_comparisons = 0
            for shape_, states_ in sorted(shape2states.items()):
                if shape == shape_:
                    continue
                for state_i in states:
                    if tuple(state_i) in [tuple(s) for s in states_]:
                        num_shared += 1
                    num_comparisons += 1
            term2 = (num_comparisons - num_shared) / num_comparisons
            # collect terms
            terms1.append(term1)
            terms2.append(term2)
            # plot
            if plot_states:
                plot_state_analysis(states_mat, title=shape)

        # compute result
        res = np.array(terms1) * np.array(terms2)
        tmp1 = np.array(terms1).round(2)
        tmp2 = np.array(terms2).round(2)
        avg_res = np.mean(res)
        is_best = ' ' if avg_res < largest_avg_res.value else 'T'
        print(f'avg={avg_res :.4f} best={is_best} rgb_id={rgb_id} {tmp1} {tmp2} h_ids={h_ids} ')

        # update shared memory
        if avg_res > largest_avg_res.value:
            largest_avg_res.value = avg_res

