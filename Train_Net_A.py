from polyomino_world.networks import dataset, network, analysis
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)
    hidden_size = 32
    hidden_actf = 'tanh'
    learning_rate = 0.20
    num_epochs = 1000
    weight_init = 0.00001
    output_freq = 25
    verbose = False
    x_type = 'WorldState' 
    y_type = 'FeatureVector' #'WorldState'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False
    processor = 'CPU'
    optimizer = 'SGD'

    training_file = 'w8-8_s9_c8_0_100_1_train_top.csv'

    test_file = 'w8-8_s9_c8_0_10_1_complete_full_actions_test.csv' 
    # 'w8-8_s9_c8_0_10_1_complete_test' 

    network_directory = 'WS_FV_2021_1_24_13_34_34_top_bottom_both_complete_test_second_stage'


    training_set = dataset.DataSet(training_file, None, included_features, processor)
    test_set = dataset.DataSet(test_file, None, included_features, processor)

    net = network.MlNet()
    # line 30 if starting a new model, line 33 if adding to an existing one
    net.init_model(x_type, y_type, training_set,
                   hidden_size, hidden_actf, optimizer, learning_rate, weight_init, processor)
    # net.load_model(network_directory, included_features, processor)

    analysis.train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate,
                     shuffle_sequences, shuffle_events, output_freq, verbose)


main()
