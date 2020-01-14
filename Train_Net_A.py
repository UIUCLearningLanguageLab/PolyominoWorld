from src.networks import dataset, network, analysis
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)

    hidden_size = 32
    learning_rate = 0.20
    num_epochs = 10
    weight_init = 0.01
    output_freq = 10
    verbose = False
    x_type = 'WorldState'
    y_type = 'FeatureVector'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False

    training_file = 'data/w6-6_s9_c8_0_10_0.csv'
    test_file = 'data/w6-6_s9_c8_0_1_0.csv'
    network_file = None

    training_set = dataset.DataSet(training_file, network_file, included_features)
    test_set = dataset.DataSet(test_file, network_file, included_features)

    net = network.MlNet(x_type, y_type, training_set, hidden_size, learning_rate, weight_init)

    analysis.train_a(net, training_set, test_set, num_epochs, learning_rate,
                     shuffle_sequences, shuffle_events, output_freq, verbose)


main()
