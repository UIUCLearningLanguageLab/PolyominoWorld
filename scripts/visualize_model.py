"""
A tkinter app that allows for interactive visualization of network weights, activations, and predictions.

To load pytorch models, Ludwig is used to search all results directories on the shared drive,
 and to retrieve only those results generated with hyper-parameters currently in params.param2requests.

To tell ludwig where to look for models,
create an environment variable "LUDWIG_MNT" that points to the path where ludwig_data is mounted on your machine.
"""

import torch
import yaml
from pathlib import Path

from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.display import Display
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default
from polyominoworld import configs

from ludwig.results import gen_param_paths


if __name__ == '__main__':

    configs.Device.gpu = False  # do not send x, and y to gpu

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            # isolated=True,
            # runs_path=Path(__file__).parent.parent / 'runs',
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # re-generate data  the way it was during training
        world = World(params)
        data = DataSet(world.generate_sequences(leftout_colors=('', ),
                                                leftout_shapes=('', ),
                                                leftout_variants='',
                                                leftout_positions=get_leftout_positions(''),
                                                ),
                       params,
                       name='re-generated')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        for path_to_net in param_path.rglob('model.pt'):

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()

            # visualize
            display = Display(data, net, params.allow_negative_x)
            display.root.mainloop()
