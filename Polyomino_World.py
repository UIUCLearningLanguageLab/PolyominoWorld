from polyomino_world.display import display
from polyomino_world.networks import dataset
from polyomino_world.networks import network
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True)
    dataset_filename = 'w8-8_s9_c8_0_10_1.csv'
    model1 = 'WS_FV_2020_2_16_12_34_34'
    model2 = 'WS_WS_2020_2_9_10_44_25'
    included_features = [1, 1, 1, 0]
    processor = 'CPU'

    the_dataset = dataset.DataSet(dataset_filename, None, included_features, processor)
    the_network = network.MlNet()
    the_network.load_model(model1, included_features, processor)

    the_display = display.Display(the_dataset, the_network)
    the_display.root.mainloop()


main()
