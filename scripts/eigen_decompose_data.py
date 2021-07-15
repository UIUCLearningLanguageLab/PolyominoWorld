"""
perform pca on data
"""

import numpy as np

from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet

from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2default


if __name__ == '__main__':

    params = Params.from_param2val(param2default)
    world = World(params)
    data = DataSet(world.generate_sequences(leftout_colors=(),
                                            leftout_shapes=(),
                                            leftout_variants='',
                                            leftout_positions=get_leftout_positions(''),
                                            ),
                   seed=params.seed,
                   shuffle_events=params.shuffle_events,
                   shuffle_sequences=params.shuffle_sequences,
                   name='re-generated')

    data_matrix = np.vstack([event.get_x('world').cpu().numpy().flatten() for event in data.get_events()])
    print(data_matrix.shape)

    u, s, v = np.linalg.svd(data_matrix)
    print(s)
