"""
A tkinter app that allows for interactive visualization of network weights, activations, and predictions.

To load pytorch models, Ludwig is used to search all results directories on the shared drive,
 and to retrieve only those results generated with hyper-parameters currently in params.param2requests.

If trained models are saved locally, use RUNS_PATH instead of LUDWIG_DATA_PATH so that Ludwig will search locally.
"""

from pathlib import Path
import torch
from typing import Optional
import yaml

from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.display import Display
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default

from ludwig.results import gen_param_paths

LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH: Optional[Path] = None  # if using local runs or None if using runs from Ludwig

if __name__ == '__main__':

    project_name = 'PolyominoWorld'
    for param_path, label in gen_param_paths(
            project_name,
            param2requests,
            param2default,
            runs_path=RUNS_PATH,
            ludwig_data_path=LUDWIG_DATA_PATH,
    ):

        # load hyper-parameter settings
        with (param_path / 'param2val.yaml').open('r') as f:
            param2val = yaml.load(f, Loader=yaml.FullLoader)
        params = Params.from_param2val(param2val)

        # re-generate data  the way it was during training
        world = World(params)
        data = DataSet(world.generate_sequences(), params, name='re-generated')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        for path_to_net in param_path.rglob('model.pt'):

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()

            # visualize
            display = Display(data, net)
            display.root.mainloop()
