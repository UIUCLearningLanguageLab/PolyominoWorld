from src import dataset
from src import numpy_ffnet
import numpy as np


def test(net, the_dataset):
    costs = np.array([0, 0, 0, 0, 0], float)
    the_dataset.create_xy(False)
    for i in range(the_dataset.x.shape[0]):
        o, h, o_cost = net.test(the_dataset.x[i], the_dataset.y[i])
        costs[0] += (o_cost**2).sum()
        costs[1] += (o_cost[:the_dataset.index_starts[0]] ** 2).sum()
        costs[2] += (o_cost[the_dataset.index_starts[0]:the_dataset.index_starts[1]] ** 2).sum()
        costs[3] += (o_cost[the_dataset.index_starts[1]:the_dataset.index_starts[2]] ** 2).sum()
        costs[4] += (o_cost[the_dataset.index_starts[2]:the_dataset.index_starts[3]] ** 2).sum()
    costs /= the_dataset.x.shape[0]
    costs /= np.array([the_dataset.y.shape[1], the_dataset.num_shapes, the_dataset.num_sizes, the_dataset.num_colors, the_dataset.num_actions], float)
    return costs


def evaluate(net, the_dataset):
    the_dataset.create_xy(False)
    for i in range(the_dataset.x.shape[0]):
        o, h, o_cost = net.test(the_dataset.x[i], the_dataset.y[i])
        shape_actual = the_dataset.y[i][:the_dataset.index_starts[0]]
        size_actual = the_dataset.y[i][the_dataset.index_starts[0]:the_dataset.index_starts[1]]
        color_actual = the_dataset.y[i][the_dataset.index_starts[1]:the_dataset.index_starts[2]]
        action_actual = the_dataset.y[i][the_dataset.index_starts[2]:the_dataset.index_starts[3]]

        shape_outputs = o[:the_dataset.index_starts[0]]
        size_outputs = o[the_dataset.index_starts[0]:the_dataset.index_starts[1]]
        color_outputs = o[the_dataset.index_starts[1]:the_dataset.index_starts[2]]
        action_outputs = o[the_dataset.index_starts[2]:the_dataset.index_starts[3]]

        shape_index = np.argmax(shape_actual)
        shape_guess = shape_outputs[shape_index]

        size_index = np.argmax(size_actual)
        size_guess = shape_outputs[size_index]
        color_index = np.argmax(color_actual)
        color_guess = shape_outputs[color_index]
        action_index = np.argmax(action_actual)
        action_guess = shape_outputs[action_index]

        print("{}  {}  {:0.3f}  {:0.3f}   {:0.3f}   {:0.3f}".format(i, the_dataset.label_list[i],
                                                                    shape_guess, size_guess, color_guess, action_guess))

        print("\t\tShape Actual:  {}".format(shape_actual))
        print("\t\tShape Guess:   {}".format(shape_outputs))
        print("\t\tSize Actual:   {}".format(size_actual))
        print("\t\tSize Guess:    {}".format(size_outputs))
        print("\t\tColor Actual:  {}".format(color_actual))
        print("\t\tColor Guess:   {}".format(color_outputs))
        print("\t\tAction Actual: {}".format(action_actual))
        print("\t\tAction Guess:  {}".format(action_outputs))


def train(net, the_dataset, num_epochs, learning_rate):
    randomize = True
    for i in range(num_epochs):
        the_dataset.create_xy(randomize)
        for j in range(the_dataset.x.shape[0]):
            net.train(the_dataset.x[j], the_dataset.y[j], learning_rate)
        costs = test(net, the_dataset)
        if i % 10 == 0:
            print("Epoch:{} {:16s} costs: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(i, the_dataset.name, costs[0], costs[1], costs[2],
                                                                            costs[3], costs[4]))

def main():

    input_size = 432
    hidden_size = 24
    output_size = 25
    learning_rate = 0.20
    num_epochs = 1000
    weight_init = [0, 0.0000001]

    training_set = dataset.Dataset('training.csv')
    test_set = dataset.Dataset('test.csv')

    net = numpy_ffnet.NumpyFfnet(input_size, hidden_size, output_size, weight_init)
    costs = test(net, training_set)
    print("{} Cost: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(training_set.name, costs[0], costs[1], costs[2],
                                                                    costs[3], costs[4]))

    train(net, training_set, num_epochs, learning_rate)
    evaluate(net, test_set)
    costs = test(net, training_set)
    print("{:16s} Cost: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(training_set.name, costs[0], costs[1], costs[2],
                                                                    costs[3], costs[4]))
    costs = test(net, test_set)
    print("{:16s} Cost: {:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(test_set.name, costs[0], costs[1], costs[2],
                                                                    costs[3], costs[4]))

main()
