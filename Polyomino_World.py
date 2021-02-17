from polyomino_world.display import display
from polyomino_world.networks import dataset
from polyomino_world.networks import network
import numpy as np


def main():
    np.set_printoptions(precision=3, suppress=True)
    dataset_filename = 'w8-8_s9_c8_location-all_10_0_allpossible_8x8_topbottom.csv' #'w8-8_s9_c8_0_20_1.csv' # w8-8_s9_c8_0_100_1_first_half_variant_train
    model1 = 'WS_FV_2021_1_25_9_29_20_top_bottom_train_top_test_full_first_stage_check'  # 'WS_FV_2020_2_16_12_34_34'
    model2 = 'WS_FV_2021_1_24_18_7_12_top_bottom_train_top_test_bottom_first_stage' # 'WS_WS_2020_2_9_10_44_25'
    included_features = [1, 1, 1, 0]
    processor = 'CPU'

    the_dataset = dataset.DataSet(dataset_filename, None, included_features, processor)
    the_network = network.MlNet()
    the_network.load_model(model1, included_features, processor)

    the_display = display.Display(the_dataset, the_network)
    the_display.root.mainloop()


main()
