"""
Plot discrete hidden states for each shape
"""
import matplotlib.pyplot as plt
import torch
import yaml
import numpy as np
from pathlib import Path

from polyominoworld import configs
from polyominoworld.figs import plot_state_analysis
from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.world import World
from polyominoworld.network import Network
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests

from ludwig.results import gen_param_paths

RGB_ID = None
HIDDEN_LAYER_ID = 0


if __name__ == '__main__':

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            isolated=True,
            runs_path=Path(__file__).parent.parent / 'runs',
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # use all locations, rotations, shapes, and colors - and filter later
        world = World(params)
        data = DataSet(world.generate_sequences(leftout_colors=('',),
                                                leftout_shapes=('',),
                                                leftout_variants='',
                                                leftout_positions=get_leftout_positions(''),
                                                ),
                       params,
                       name='re-generated')

        if RGB_ID is not None:
            name_of_color_channel = {0: 'red', 1: 'green', 2: 'blue'}[RGB_ID]
        else:
            name_of_color_channel = None

        # multiple models may exist for the same hyper-parameter configuration - use first only
        for path_to_net in list(param_path.rglob('model.pt'))[:1]:

            print(f'Loading net from {path_to_net}')

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()
            h_x = net.h_xs[HIDDEN_LAYER_ID].weight.detach().numpy()  # [num hidden, num world cells]

            # get set of detectors for one or all color channel
            detectors = []
            for h_id, hi in enumerate(h_x):
                # get weights to one color or all channel
                if RGB_ID is not None:
                    hi_reshaped = hi.reshape((3, configs.World.max_x, configs.World.max_y))[RGB_ID, :, :]
                else:
                    hi_reshaped = hi
                # scale so that modes of hi are aligned with integers in base 1
                hi_scaled = hi_reshaped * 1.0
                # make discrete
                hi_discrete = np.rint(hi_scaled)
                # collect detectors only if they are supposed to be collected
                detectors.append(hi_discrete.flatten())
            detector_mat = np.array(detectors)

            # compute states produced by dot product of input and detectors
            shape2states = {shape: [] for shape in configs.World.master_shapes}
            shape2states_other = {shape: [] for shape in configs.World.master_shapes}
            for event in data.get_events():
                # filter by color channel
                if RGB_ID is not None and event.color != name_of_color_channel:
                    continue
                # get input from single color channel
                if RGB_ID is not None:
                    x = event.world_vector.as_3d()[RGB_ID].flatten()
                else:
                    x = event.world_vector.vector.numpy()
                # compute discrete state
                state = tuple(detector_mat @ x)
                # collect states
                shape2states[event.shape].append(state)

            for shape, states in shape2states.items():
                plot_state_analysis(np.array(states),
                                    title=f'shape={shape}\ncolor channel={RGB_ID}',
                                    )

        raise SystemExit  # do not keep searching for models - regular pattern ids are defined for first model only
