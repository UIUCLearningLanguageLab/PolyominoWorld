from polyomino_world.networks import dataset, network, analysis
import pandas as pd
import sys
import numpy as np


def main(param2val):

    print(param2val, flush=True)

    training_file = param2val['training_file']
    test_file = param2val['test_file']
    network_file = param2val['network_file']
    included_features = param2val['included_features']

    x_type = param2val['x_type']
    y_type = param2val['y_type']
    hidden_size = param2val['hidden_size']
    learning_rate = param2val['learning_rate']
    weight_init = param2val['weight_init']
    num_epochs = param2val['num_epochs']
    shuffle_sequences = param2val['shuffle_sequences']
    shuffle_events = param2val['shuffle_events']
    output_freq = param2val['output_freq']
    verbose = param2val['verbose']

    save_path = param2val['save_path']  # TODO use this to save any files into this directory
    project_path = param2val['project_path']  # TODO use this to save any files into this directory

    training_set = dataset.DataSet(training_file, network_file, included_features, project_path)
    test_set = dataset.DataSet(test_file, network_file, included_features, project_path)

    sys.stdout.flush()

    net = network.MlNet(x_type, y_type, training_set, hidden_size, learning_rate, weight_init, project_path)
    net.cuda()

    results_dict = analysis.train_a(net, training_set, test_set, num_epochs, learning_rate,
                                    shuffle_sequences, shuffle_events, output_freq, verbose)

    sys.stdout.flush()
    eval_epochs = results_dict['epoch']
    series_list = []
    for k, v in results_dict.items():
        if not np.any(np.isnan(v)):
            s = pd.Series(v, index=eval_epochs)
        else:
            continue
        s.name = k
        series_list.append(s)

    print('Done with job.main', flush=True)

    return series_list


    # TODO add stage 2 training here