from src import dataset
from src import numpy_ffnet


def main():

    input_size = 300
    hidden_size = 64
    output_size = 21
    learning_rate = 0.3
    num_epochs = 1
    weight_init = [0, 0.0000001]

    training_set = dataset.Dataset('training.csv')
    # test_set = dataset.Dataset('test.csv')

    # net = numpy_ffnet.NumpyFfnet(input_size, hidden_size, output_size, weight_init)

    # train_cost = net.test(training_set.x, training_set.y)
    # test_cost = net.test(test_set.x, test_set.y)
    # print("Start Train Cost: {:0.3f}   Start Test Cost: {:0.3f}".format(train_cost, test_cost))

    # for i in range(num_epochs):
    #     training_set.create_xy(True)
    #     net.train(training_set.x, training_set.y, learning_rate)
    #     if i % 10 == 0:
    #         train_cost = net.test(training_set.x, training_set.y)
    #         test_cost = net.test(test_set.x, test_set.y)
    #         print("Epoch: {}    Train Cost: {:0.3f}   Test Cost: {:0.3f}".format(i, train_cost, test_cost))
    #
    # train_cost = net.test(training_set.x, training_set.y)
    # test_cost = net.test(test_set.x, test_set.y)
    # print("Final Train Cost: {:0.3f}   Final Test Cost: {:0.3f}".format(train_cost, test_cost))


main()
