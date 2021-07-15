"""
Save vectors of hidden activations for each data instance to a tsv file.
Additionally, save metadata describing shape, size, color of instance to second tsv file.

For use with https://projector.tensorflow.org/
"""

import torch
import yaml
from pathlib import Path
import pandas as pd
import numpy as np

from polyominoworld import configs
from polyominoworld.params import Params
from polyominoworld.params import param2default, param2requests
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.world import World
from polyominoworld.utils import get_leftout_positions

from ludwig.results import gen_param_paths

HIDDEN_LAYER_ID = 0


if __name__ == '__main__':

    configs.Device.gpu = False

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
        params: Params = Params.from_param2val(param2val)

        # re-generate data  the way it was during training
        world = World(params)
        dataset = DataSet(world.generate_sequences(leftout_colors=params.train_leftout_colors,
                                                   leftout_shapes=params.train_leftout_shapes,
                                                   leftout_variants=params.train_leftout_variants,
                                                   leftout_positions=get_leftout_positions(params.train_leftout_half),
                                                   ),
                          seed=params.seed,
                          shuffle_events=params.shuffle_events,
                          shuffle_sequences=params.shuffle_sequences,
                          name='re-generated')

        # multiple models may exist for the same hyper-parameter configuration
        for path_to_net in list(param_path.rglob('model.pt')):

            print(f'Loading net from {path_to_net}')

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.requires_grad_(False)
            net.eval()

            header2column = {
                'shape': [],
                'size': [],
                'color': [],
            }
            vectors = []
            for event in dataset.get_events():

                x = event.get_x(net.params.x_type)
                hs = [h.numpy() for h in net.compute_hs(x)]
                h = hs[HIDDEN_LAYER_ID]
                # collect
                vectors.append(h)
                header2column['shape'].append(event.shape)
                header2column['size'].append(event.size)
                header2column['color'].append(event.color)

            # make tsv file
            df1 = pd.DataFrame(data=np.array(vectors))
            df2 = pd.DataFrame(data=header2column)
            df1.to_csv(configs.Dirs.embedding_projector / f'{param_path.name}_vectors.tsv',
                       index=False, sep='\t', header=False)
            df2.to_csv(configs.Dirs.embedding_projector / f'{param_path.name}_metadata.tsv',
                       index=False, sep='\t')
            print(f'Saved tsv files to {configs.Dirs.embedding_projector}')

            break  # save tsv file for first  replication only

        else:
            raise RuntimeError(f'Did not find any saved models in {param_path}')

        raise SystemExit  # do not keep searching for models
