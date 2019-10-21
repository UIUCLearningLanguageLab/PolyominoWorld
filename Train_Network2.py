from src.networks import pytorch_nets, dataset
import numpy as np


def main():
    np.set_printoptions(precision=4, suppress=True)

    hidden_size = 16
    learning_rate = 0.20
    num_epochs = 1000
    weight_init = 0.0000001
    output_freq = 100
    verbose = False

    training_set = dataset.Dataset('data/w6-6_s9_c8_0_100_2.csv')
    test_set = dataset.Dataset('data/w6-6_s9_c8_0_5_0.csv')

    net = pytorch_nets.FFNet(training_set.x_size, hidden_size, training_set.y_size, weight_init)
    costs = pytorch_nets.test(net, training_set)
    print("{:14s} Cost: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(training_set.name, costs[0], costs[1], costs[2], costs[3]))

    # analyses.train(net, training_set, test_set, num_epochs, learning_rate, output_freq)
    #
    # analyses.evaluate(net, test_set, verbose)
    #
    # analyses.evaluate(net, training_set, verbose)
    # analyses.evaluate(net, test_set, verbose)
    #
    # filename = "models/nff_h{}_lr{}_e{}.csv".format(hidden_size, str(learning_rate)[2:], num_epochs)
    # net.save_weights(filename)


main()
