import itertools
from typing import List, Tuple

from polyominoworld import configs


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


def calc_world_vector_size(add_grayscale: bool,
                           ) -> int:
    num_cells = configs.World.max_x * configs.World.max_x
    if add_grayscale:
        num_channels = 4
    else:
        num_channels = 3  # rgb
    return num_cells * num_channels