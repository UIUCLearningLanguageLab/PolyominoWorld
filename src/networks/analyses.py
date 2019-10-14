import numpy as np


def train(net, training_set, test_set, num_epochs, learning_rate, output_freq):
    randomize = True
    for i in range(num_epochs):
        training_set.create_xy(randomize)
        for j in range(training_set.x.shape[0]):
            net.train(training_set.x[j], training_set.y[j], learning_rate)
        costs = test(net, training_set)
        if (i+1) % output_freq == 0:
            #print("Epoch:{} {:16s} costs: {:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(i, training_set.name,
                                                                                  # costs[0], costs[1],
                                                                                  # costs[2], costs[3]))
            evaluate(net, training_set, False)
            evaluate(net, test_set, False)
            print('\n')


def test(net, the_dataset):
    costs = np.array([0, 0, 0, 0], float)
    the_dataset.create_xy(False)
    for i in range(the_dataset.x.shape[0]):
        o, h, o_cost = net.test(the_dataset.x[i], the_dataset.y[i])
        costs[0] += (o_cost[:the_dataset.index_starts[0]] ** 2).sum()
        costs[1] += (o_cost[the_dataset.index_starts[0]:the_dataset.index_starts[1]] ** 2).sum()
        costs[2] += (o_cost[the_dataset.index_starts[1]:the_dataset.index_starts[2]] ** 2).sum()
        costs[3] += (o_cost[the_dataset.index_starts[2]:the_dataset.index_starts[3]] ** 2).sum()
    costs /= the_dataset.x.shape[0]
    costs /= np.array([the_dataset.num_shapes_all,
                       the_dataset.num_sizes_all,
                       the_dataset.num_colors_all,
                       the_dataset.num_actions_all], float)
    return costs


def evaluate(net, the_dataset, verbose):
    shuffle = False
    the_dataset.create_xy(shuffle)

    accuracy_array = np.zeros([4], float)
    costs = np.zeros([4], float)

    for i in range(the_dataset.x.shape[0]):
        o, h, o_cost = net.test(the_dataset.x[i], the_dataset.y[i])

        costs[0] += (o_cost[:the_dataset.index_starts[0]] ** 2).sum()
        shape_actual = the_dataset.y[i][:the_dataset.index_starts[0]]
        shape_outputs = o[:the_dataset.index_starts[0]]
        shape_actual_index = np.argmax(shape_actual)
        shape_actual_score = shape_outputs[shape_actual_index]
        shape_guess_index = np.argmax(shape_outputs)
        if shape_actual_index == shape_guess_index:
            accuracy_array[0] += 1

        costs[1] += (o_cost[the_dataset.index_starts[0]:the_dataset.index_starts[1]] ** 2).sum()
        size_actual = the_dataset.y[i][the_dataset.index_starts[0]:the_dataset.index_starts[1]]
        size_outputs = o[the_dataset.index_starts[0]:the_dataset.index_starts[1]]
        size_actual_index = np.argmax(size_actual)
        size_actual_score = size_outputs[size_actual_index]
        size_guess_index = np.argmax(size_outputs)
        if size_actual_index == size_guess_index:
            accuracy_array[1] += 1

        costs[2] += (o_cost[the_dataset.index_starts[1]:the_dataset.index_starts[2]] ** 2).sum()
        color_actual = the_dataset.y[i][the_dataset.index_starts[1]:the_dataset.index_starts[2]]
        color_outputs = o[the_dataset.index_starts[1]:the_dataset.index_starts[2]]
        color_actual_index = np.argmax(color_actual)
        color_actual_score = color_outputs[color_actual_index]
        color_guess_index = np.argmax(color_outputs)
        if color_actual_index == color_guess_index:
            accuracy_array[2] += 1

        costs[3] += (o_cost[the_dataset.index_starts[2]:the_dataset.index_starts[3]] ** 2).sum()
        action_actual = the_dataset.y[i][the_dataset.index_starts[2]:the_dataset.index_starts[3]]
        action_outputs = o[the_dataset.index_starts[2]:the_dataset.index_starts[3]]
        action_actual_index = np.argmax(action_actual)
        action_actual_score = action_outputs[action_actual_index]
        action_guess_index = np.argmax(action_outputs)
        if action_actual_index == action_guess_index:
            accuracy_array[3] += 1

        if verbose:
            print("{}  {}  {:0.3f}  {:0.3f}   {:0.3f}   {:0.3f}".format(i, the_dataset.label_list[i],
                                                                        shape_actual_score, size_actual_score,
                                                                        color_actual_score, action_actual_score))

            print("\t\tShape Actual:  {}".format(shape_actual))
            print("\t\tShape Guess:   {}".format(shape_outputs))
            print("\t\tSize Actual:   {}".format(size_actual))
            print("\t\tSize Guess:    {}".format(size_outputs))
            print("\t\tColor Actual:  {}".format(color_actual))
            print("\t\tColor Guess:   {}".format(color_outputs))
            print("\t\tAction Actual: {}".format(action_actual))
            print("\t\tAction Guess:  {}".format(action_outputs))

    costs /= the_dataset.x.shape[0]
    costs /= np.array([the_dataset.num_shapes_all,
                       the_dataset.num_sizes_all,
                       the_dataset.num_colors_all,
                       the_dataset.num_actions_all], float)
    pc = accuracy_array / the_dataset.x.shape[0]
    print("\t{:20s}  |  {:0.2f}  {:0.2f}  {:0.2f}  {:0.2f}  |  {:0.2f}  {:0.2f}  {:0.2f}  {:0.2f}".format(the_dataset.name,
                                                            costs[0], costs[1], costs[2], costs[3],
                                                            pc[0]*100, pc[1]*100, pc[2]*100, pc[3]*100))