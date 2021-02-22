"""A tkinter app that allows for interactive visualization of network weights, activations, and predictions"""

import numpy as np

from polyominoworld.dataset import DataSet
from polyominoworld.network import Network
from polyominoworld.display import Display
from polyominoworld.world import World
from polyominoworld.params import Params


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)

    # todo
    params = Params.from_trained()
    world = World(params)
    data = DataSet(world, params)
    net = Network(params)

    # visualize
    display = Display(data, net)
    display.root.mainloop()