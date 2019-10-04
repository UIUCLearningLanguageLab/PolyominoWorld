from src import dataset
from src import numpy_ffnet
from src import config


def main():

    input_size = 300
    hidden_size = 64
    output_size = 21
    learning_rate = 0.3
    num_epochs = 1
    weight_init = [0, 0.0000001]
    batch_size = 3600

    training_set = dataset.Dataset('training.csv')
    test_set = dataset.Dataset('test.csv')

    net = numpy_ffnet.NumpyFfnet(input_size, hidden_size, output_size, weight_init)
    for i in range(num_epochs):
        x, y, labels, events, turns = training_set.batch_shuffled_data(batch_size)
        for j in range(len(x)):
            print(j, events[j], turns[j], labels[j], y[j])

    #     net.train(dataset, learning_rate)
    #
    # net.test(test_set)


main()
