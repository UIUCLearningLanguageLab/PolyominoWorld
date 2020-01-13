from src.networks import dataset, network, analysis
import numpy as np
import sys


def main():
    np.set_printoptions(precision=4, suppress=True)

    hidden_size = 32
    learning_rate = 0.25
    num_epochs = 20
    weight_init = 0.000001
    output_freq = 5
    verbose = True
    net_type = 'classifier'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False

    training_file = 'data/w6-6_s9_c8_0_100_0.csv'
    test_file = 'data/w6-6_s9_c8_0_10_0.csv'
    network_file = None

    training_set = dataset.DataSet("Train_72_W100_0", training_file, network_file, included_features)
    test_set = dataset.DataSet("Test_72_W100_0", test_file, network_file, included_features)

    if net_type == 'autoassociator':
        output_size = training_set.world_size
    elif net_type == 'classifier':
        output_size = training_set.num_included_features
    else:
        print("Net Type {} not recognized")
        sys.exit()

    net = network.MlNet(net_type, training_set.world_size, hidden_size, output_size, weight_init)

    analysis.train_a(net, training_set, test_set,
                     num_epochs, learning_rate,
                     shuffle_sequences, shuffle_events,
                     output_freq, verbose)


main()
