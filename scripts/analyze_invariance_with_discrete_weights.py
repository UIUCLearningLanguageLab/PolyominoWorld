"""
Experimental script to quantify how invariant the hidden states are to
rotation and location variants of a shape.

1) Input weights are first filtered based on whether they exhibit a regular pattern,
and then made discrete by rounding to the nearest mode.
2) The hidden state is computed using these filtered, discrete weights.
3) a measure of invariance is computed using two terms:
    a) proportion of states states shared within a shape
    b) proportion of states not shared between shapes

Notes:
    - each primary color channel is evaluated separately. all combinations of input weight patterns are searched,
    to identify the combination of patterns that results in highest measure of invariance.
    - measure of invariance is a product of two proportions, consequently max=1 and min=0

"""

import torch
import yaml
import numpy as np
from itertools import combinations


from polyominoworld import configs
from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default
from polyominoworld.figs import plot_hidden_weights_analysis, plot_state_analysis

from ludwig.results import gen_param_paths

SCALE = 1.1  # scale weights so that rounding to nearest integer effectively rounds to nearest mode

# manually specify ids of input weights that appear regular. do this for first model replication, which is loaded first
REGULAR_PATTERN_IDS = [3, 4, 5, 6, 7, 11, 12, 14, 16, 20, 22, 23, 24, 25, 29, 30, 31][::-1]
# REGULAR_PATTERN_IDS = [25, 30, 31]
# REGULAR_PATTERN_IDS = [6, 11]
PLOT_HI = False
PLOT_STATES = False

if __name__ == '__main__':

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # only use one color channel
        for rgb_id, color in enumerate(['red', 'green', 'blue']):

            largest_avg_res = 0  # this stores the largest result identified during search

            # use all locations, rotations, and shapes, but only one primary color
            world = World(params)
            data = DataSet(world.generate_sequences(leftout_colors=tuple([c for c in configs.World.master_colors
                                                                          if c != color]),
                                                    leftout_shapes=('', ),
                                                    leftout_variants='',
                                                    leftout_positions=get_leftout_positions(''),
                                                    ),
                           params,
                           name='re-generated')

            # multiple models may exist for the same hyper-parameter configuration - iterate over each
            for path_to_net in list(param_path.rglob('model.pt'))[:1]:

                print(f'Loading net from {path_to_net}')

                # load net
                net = Network(params)
                state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
                net.load_state_dict(state_dict)
                net.eval()

                h_x = net.h_x.weight.detach().numpy()  # [num hidden, num world cells]

                # for some combination of input weight patterns
                for combo_size in range(1, len(REGULAR_PATTERN_IDS) + 1):
                    for h_ids in combinations(REGULAR_PATTERN_IDS, combo_size):

                        # for each hidden weight pattern, save it if it is regular
                        detectors = []
                        for h_id, hi in enumerate(h_x):
                            # get weights to first color channel only (red)
                            hi_single_channel = hi.reshape((3, configs.World.max_x, configs.World.max_y))[rgb_id, :, :]
                            # scale so that modes of hi are aligned with integers in base 1
                            hi_single_channel_scaled = hi_single_channel * SCALE
                            # make discrete
                            hi_single_channel_discrete = np.rint(hi_single_channel_scaled)
                            if h_id in h_ids:
                                detectors.append(hi_single_channel_discrete.flatten())

                            if PLOT_HI:
                                x_tick_labels = [f'x{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[0])]
                                y_tick_labels = [f'y{i + 1:0>2}' for i in range(hi_single_channel_scaled.shape[1])]
                                plot_hidden_weights_analysis(hi_single_channel_scaled,
                                                             title=f'pattern{h_id:0>3}\nregular={h_id in h_ids}',
                                                             x_tick_labels=x_tick_labels,
                                                             y_tick_labels=y_tick_labels,
                                                             )

                        detector_mat = np.array(detectors)

                        # compute states produced by dot product of input and detectors
                        shape2states = {shape: [] for shape in configs.World.master_shapes}
                        for event in data.get_events():
                            # get input from single color channel
                            x = event.world_vector.as_3d()[rgb_id].flatten()

                            # compute discrete state
                            state = detector_mat @ x
                            shape2states[event.shape].append(state)

                        # analyze the invariance to rotation and location, for each shape separately
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

                            if PLOT_STATES:
                                plot_state_analysis(states_mat, title=shape)

                        # report average
                        res = np.array(terms1) * np.array(terms2)
                        tmp1 = np.array(terms1).round(2)
                        tmp2 = np.array(terms2).round(2)
                        avg_res = np.mean(res)
                        is_best = ' ' if avg_res < largest_avg_res else 'T'
                        print(f'avg={avg_res :.4f} best={is_best} color={color} {tmp1} {tmp2} h_ids={h_ids} ')

                        if avg_res > largest_avg_res:
                            largest_avg_res = avg_res

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
