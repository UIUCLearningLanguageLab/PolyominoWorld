"""
evaluate a model, after it has been trained (offline)

currently implemented:
 - EVAL_LEFTOUT_VARIANTS: compute average cost by shape, on variants not seen during training
"""

import torch
import yaml
import numpy as np

from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld import configs
from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default

from ludwig.results import gen_param_paths


EVAL_LEFTOUT_VARIANTS = True

param2requests = {
    'train_leftout_variants': ['half1'],
}

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
        params: Params = Params.from_param2val(param2val)

        # are any features leftout?  # TODO add option to leave out other features
        if EVAL_LEFTOUT_VARIANTS:
            leftout_variants_inverse = {'half1': 'half2', 'half2': 'half1'}[params.train_leftout_variants]
        else:
            leftout_variants_inverse = None

        # re-generate data  the way it was during training
        world = World(params)
        dataset = DataSet(world.generate_sequences(leftout_colors=params.train_leftout_colors,
                                                   leftout_shapes=params.train_leftout_shapes,
                                                   leftout_variants=leftout_variants_inverse or params.train_leftout_variants,
                                                   leftout_positions=get_leftout_positions(params.train_leftout_half),
                                                   ),
                          seed=params.seed,
                          shuffle_events=params.shuffle_events,
                          shuffle_sequences=params.shuffle_sequences,
                          name='re-generated')

        # loss function
        if params.criterion == 'mse':
            if params.y_type == 'world':  # use MSE only with auto-associator
                raise RuntimeError('MSE loss should only be used with auto-associator')
            else:
                criterion_avg = torch.nn.MSELoss()
                criterion_all = torch.nn.MSELoss(reduction='none')  # for evaluation
        elif params.criterion == 'bce':
            criterion_avg = torch.nn.BCEWithLogitsLoss()  # sigmoid function and then a binary cross entropy
            criterion_all = torch.nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise AttributeError(f'Invalid arg to criterion')

        # multiple models may exist for the same hyper-parameter configuration - iterate over each
        for path_to_net in param_path.rglob('model.pt'):

            print()
            print(f'Evaluating model={path_to_net}')

            # load net
            net = Network(params)
            state_dict = torch.load(path_to_net, map_location=torch.device('cpu'))
            net.load_state_dict(state_dict)
            net.eval()

            cost_by_shape = np.zeros((len(configs.World.master_shapes), 1))
            num_by_shape = np.zeros((len(configs.World.master_shapes), 1))

            shape2id = {s: n for n, s in enumerate(configs.World.master_shapes)}

            # evaluate
            for event in dataset.get_events():

                x = event.get_x(net.params.x_type)
                y = event.get_y(net.params.y_type)

                o = net.forward(x)
                costs_by_feature = criterion_all(o, y).detach().cpu().numpy()
                o = o.detach().cpu().numpy()

                # collect cost for shape only
                for n, feature_label in enumerate(event.feature_vector.get_feature_labels()):

                    tmp = feature_label.value_name
                    try:
                        i = shape2id[tmp]
                    except KeyError:
                        continue
                    else:
                        cost_by_shape[i] += costs_by_feature[n]
                        num_by_shape[i] += 1

            cost_by_shape = cost_by_shape / num_by_shape
            for shape, row in zip(configs.World.master_shapes, cost_by_shape):
                print(f'shape={shape:<24} avg cost={row.item():.4f}')


