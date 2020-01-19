from polyomino_world.networks import dataset, network, analysis
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)
    hidden_size = 32
    learning_rate = 0.05
    num_epochs = 500
    weight_init = 0.001
    output_freq = 10
    verbose = False
    x_type = 'WorldState'
    y_type = 'FeatureVector'
    included_features = [1, 1, 1, 0]  # Include: Shape, Size, Color, Action
    shuffle_sequences = True
    shuffle_events = False
    processor = 'CPU'
    optimizer = 'Adam'

    project_path = './'
    training_file = 'w6-6_s9_c8_0_100_0.csv'
    test_file = 'w6-6_s9_c8_0_10_0.csv'
    network_file = None

    training_set = dataset.DataSet(training_file, network_file, included_features, project_path, processor)
    test_set = dataset.DataSet(test_file, network_file, included_features, project_path, processor)

    net = network.MlNet()
    net.init_model(x_type, y_type, training_set, hidden_size, optimizer, learning_rate, weight_init,
                        project_path, processor)

    results_dict = analysis.train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate,
                                    shuffle_sequences, shuffle_events, output_freq, verbose)

    print('Done with job.main', flush=True)


main()
