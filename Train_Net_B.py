from polyomino_world.networks import dataset, network, analysis
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)

    learning_rate = 0.20
    num_epochs = 5000
    weight_init = 0.01
    output_freq = 10
    verbose = False
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False

    training_file = 'data/w6-6_s9_c8_0_1_0.csv'
    test_file = 'data/w6-6_s9_c8_0_1_0.csv'
    network_file = 'models/classifier_2020_1_13_21_23_12/states_e100.csv'

    training_set = dataset.DataSet(training_file, network_file, included_features)
    test_set = dataset.DataSet(test_file, network_file, included_features)

    net = network.SlNet(training_set, learning_rate, weight_init)

    analysis.train_b(net, training_set, test_set, num_epochs, learning_rate,
                     shuffle_sequences, shuffle_events, output_freq, verbose)


main()
