from src.networks import network
import numpy as np
import torch


def main():
    np.set_printoptions(precision=4, suppress=True)

    learning_rate = 0.20
    num_epochs = 100
    weight_init = 0.0000001
    output_freq = 5
    verbose = False
    net_type = 'autoassociator'

    network_file = 'models/i192-h16-o25-ff.csv'

    net = network.SlNet(network_file, learning_rate)

    for i in range(num_epochs):
        epoch_loss = 0
        for j in range(net.num_items):
            out, loss = net.train_item(net.hidden[j], net.y[j])
            epoch_loss += loss.sum()


main()
