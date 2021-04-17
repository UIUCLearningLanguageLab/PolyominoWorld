"""
Count number of times each shape, or color etc. appears in a dataset
"""

from collections import defaultdict
import yaml
from pathlib import Path

from polyominoworld.utils import get_leftout_positions
from polyominoworld.dataset import DataSet

from polyominoworld.world import World
from polyominoworld.params import Params
from polyominoworld.params import param2requests, param2default

from ludwig.results import gen_param_paths


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

        # re-generate data  the way it was during training
        world = World(params)
        data = DataSet(world.generate_sequences(leftout_colors=('', ),
                                                leftout_shapes=('', ),
                                                leftout_variants='',
                                                leftout_positions=get_leftout_positions(''),
                                                ),
                       params,
                       name='re-generated')

        # count
        shape2f = defaultdict(lambda: 0)
        for event in data.get_events():
            shape2f[event.shape] += 1

        # report
        for shape, f in sorted(shape2f.items(), key=lambda i: i[1]):
            print(f'shape={shape:<12} frequency={f:>12,}')
