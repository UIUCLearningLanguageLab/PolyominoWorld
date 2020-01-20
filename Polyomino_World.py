from polyomino_world.display import display
from polyomino_world.networks import dataset
from polyomino_world.networks import network
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True)
    dataset_filename = 'w6-6_s9_c8_0_10_0.csv'
    # model = 'WorldState_FeatureVector_2020_1_18_14_36_42'
    model = 'WS_WS_2020_1_20_17_8_14'

    the_dataset = dataset.DataSet(dataset_filename, None, [1, 1, 1, 0], './', 'CPU')
    the_network = network.MlNet()
    the_network.load_model(model, the_dataset)

    the_display = display.Display(the_dataset, the_network)
    the_display.root.mainloop()


main()
