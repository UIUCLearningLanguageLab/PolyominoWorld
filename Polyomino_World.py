from polyomino_world.display import display
from polyomino_world.networks import dataset
from polyomino_world.networks import network
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True)
    dataset_filename = 'w8-8_s9_c8_0_10_0.csv'
    model = 'WS_WS_2020_2_9_12_9_14'
    included_features = [1, 1, 1, 0]
    processor = 'CPU'

    the_dataset = dataset.DataSet(dataset_filename, None, included_features, processor)
    the_network = network.MlNet()
    the_network.load_model(model, included_features, processor)

    the_display = display.Display(the_dataset, the_network)
    the_display.root.mainloop()


main()
