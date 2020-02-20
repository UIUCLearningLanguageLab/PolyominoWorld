import numpy as np
import torch
import time
import sys
import random
import math


def train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate, shuffle_sequences, shuffle_events,
            output_freq, verbose):

    print("Training {}-{} on {} epochs".format(net.x_type, net.y_type, num_epochs))
    training_set.create_xy(net, False, False)
    test_set.create_xy(net, False, False)
    net.performance_list = [net.current_epoch, 0]
    evaluate_network(net, training_set, test_set, False)
    net.save_network_performance()

    start_time = time.time()
    for i in range(num_epochs):

        net.current_epoch += 1

        training_set.create_xy(net, shuffle_sequences, shuffle_events)
        test_set.create_xy(net, shuffle_sequences, shuffle_events)

        if optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        if net.processor == 'GPU':
            if torch.cuda.is_available():
                for j in range(len(training_set.x)):
                    net.train_item(training_set.x[j].cuda(0), training_set.y[j].cuda(0), optimizer)
            else:
                print("GPU ERROR")
        else:
            for j in range(len(training_set.x)):
                net.train_item(training_set.x[j], training_set.y[j], optimizer)

        if net.current_epoch % output_freq == 0:
            took = time.time() - start_time
            net.training_time += took
            net.performance_list = [net.current_epoch, took/output_freq]
            start_time = time.time()
            evaluate_network(net, training_set, test_set, verbose)
            net.save_network_properties()
            net.save_network_weights()
            net.save_network_performance()

    net.save_network_states(training_set)


def evaluate_network(net, training_set, test_set, verbose):

    if net.y_type == 'FeatureVector':
        training_accuracies, training_costs, training_detailed_accuracies = evaluate_classifier_dataset(net, training_set, verbose)
        test_accuracies, test_costs, test_detailed_accuracies = evaluate_classifier_dataset(net, test_set, verbose)
    elif net.y_type == 'WorldState':
        training_accuracies, training_costs, training_detailed_accuracies = evaluate_autoassociator_dataset(net, training_set, verbose)
        test_accuracies, test_costs, test_detailed_accuracies = evaluate_autoassociator_dataset(net, test_set, verbose)
    else:
        print("Network Type not recognized")
        sys.exit()

    net.performance_list.append(training_costs.sum())
    net.performance_list.append(test_costs.sum())
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(training_costs[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(test_costs[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(training_accuracies[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(test_accuracies[i])
    for i in range(len(training_detailed_accuracies)):
        net.performance_list.append(training_detailed_accuracies[i])
    for i in range(len(test_detailed_accuracies)):
        net.performance_list.append(test_detailed_accuracies[i])

    output_string = "Epoch:{} |".format(net.current_epoch)
    for j in range(training_set.num_included_feature_types):
        output_string += "  {:0.2f}-{:0.2f}".format(training_costs[j], test_costs[j])
    output_string += "  | "
    for j in range(training_set.num_included_feature_types):
        output_string += "  {:0.2f}-{:0.2f}".format(training_accuracies[j], test_accuracies[j])
    output_string += " took:{:0.2f}s.".format(net.performance_list[1])
    print(output_string)


def evaluate_autoassociator_dataset(net, dataset, verbose):

    # initialize the matrices for keeping track of cost and accuracy
    accuracies = np.zeros([dataset.num_included_feature_types], float)
    costs = np.zeros([dataset.num_included_feature_types], float)
    detailed_accuracies = np.zeros([dataset.num_included_features], float)
    detailed_counts = np.zeros([dataset.num_included_features], float)

    # for each item in the dataset
    for i in range(len(dataset.x)):
        # run the item through the network, getting cost and network activation values
        o, h, o_cost = net.test_item(dataset.x[i], dataset.y[i])
        o_vector = o.detach().cpu().numpy()
        label_list = dataset.label_list[i]
        current_sum_cost = (o_cost ** 2).sum()
        world_size = int(len(o_vector)/3)

        # use the output vector to calculate which cells are "on", and what color they are
        color_guess_list = []
        position_guess_list = []
        min_x = dataset.num_rows + 1
        min_y = dataset.num_columns + 1
        for j in range(world_size):
            guess_rgb = np.array([o_vector[j], o_vector[j + world_size], o_vector[j + world_size * 2]])
            distance_vector = np.linalg.norm(dataset.all_color_rgb_matrix - guess_rgb, axis=1)
            guess_index = np.argmin(distance_vector)
            guess_label = dataset.all_color_label_list[guess_index]
            if guess_label != 'grey':
                # todo this may need to be num_columns, not num_rows, depending on how rXc matrix is flattened
                coordinates = [math.floor(j / dataset.num_rows), j % dataset.num_rows]
                if coordinates[0] < min_x:
                    min_x = coordinates[0]
                if coordinates[1] < min_y:
                    min_y = coordinates[1]
                color_guess_list.append(guess_label)
                position_guess_list.append(coordinates)

        # convert the list of positions of cells that are "on", to a set of top-left-aligned active coordinates
        for j in range(len(position_guess_list)):
            position_guess_list[j] = [position_guess_list[j][0]-min_x, position_guess_list[j][1] - min_y]
        position_guess_set = set(tuple(tuple(x) for x in position_guess_list))

        # check to see if the set of top-lef- aligned active coordinates matches the definition of any shapes
        shape_guess_label = None
        for j in range(len(dataset.master_shape_label_list)):
            if dataset.master_shape_position_list[j] == position_guess_set:
                shape_guess_label = dataset.master_shape_label_list[j]
                break
        if shape_guess_label is None:
            shape_guess_label = random.choice(dataset.master_shape_list)

        if len(color_guess_list) > 0:
            size_guess = len(color_guess_list)
        else:
            size_guess = random.choice([1, 2, 3, 4])

        if len(color_guess_list) == 0:
            color_guess_list = [random.choice(dataset.master_color_label_list)]
        color_guess_label = random.choice(color_guess_list)

        # print()
        # print(i, label_list)
        # print(position_guess_set, shape_guess_label)
        # print(len(color_guess_list), size_guess)
        # print(color_guess_list, color_guess_label)

        for j in range(dataset.num_included_feature_types):
            feature_type = dataset.included_feature_type_list[j]
            actual_label = label_list[j]
            feature_index = dataset.included_feature_index_dict[actual_label]
            detailed_counts[j] += 1
            costs[j] += current_sum_cost

            if feature_type == 'size':
                guess_label = size_guess
            elif feature_type == 'shape':
                guess_label = shape_guess_label
            elif feature_type == 'color':
                guess_label = color_guess_label
            elif feature_type == 'action':
                guess_label = None
            else:
                print("Feature Type Not recognized While Evaluating Autoassociator")
                sys.exit()

            if actual_label == guess_label:
                accuracies[j] += 1
                detailed_accuracies[feature_index] += 1

    costs /= len(dataset.x)
    accuracies = accuracies / len(dataset.x)

    for i in range(len(detailed_accuracies)):
        if detailed_counts[i] != 0:
            detailed_accuracies[i] = detailed_accuracies[i] / detailed_counts[i]
        else:
            detailed_accuracies[i] = np.nan

    return accuracies, costs, detailed_accuracies


def evaluate_classifier(net, training_set, test_set, verbose):

    training_accuracies, training_costs, training_detailed_accuracies = evaluate_classifier_dataset(net, training_set, verbose)
    test_accuracies, test_costs, test_detailed_accuracies = evaluate_classifier_dataset(net, test_set, verbose)

    output_string = "Epoch:{} |".format(net.current_epoch)
    for j in range(training_set.num_included_feature_types):
        output_string += "  {:0.2f}-{:0.2f}".format(training_costs[j], test_costs[j])
    output_string += "  | "
    for j in range(training_set.num_included_feature_types):
        output_string += "  {:0.2f}-{:0.2f}".format(training_accuracies[j], test_accuracies[j])
    output_string += " took:{:0.2f}s.".format(net.performance_list[1])
    print(output_string)

    net.performance_list.append(training_costs.sum())
    net.performance_list.append(test_costs.sum())
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(training_costs[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(test_costs[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(training_accuracies[i])
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(test_accuracies[i])
    for i in range(len(training_detailed_accuracies)):
        net.performance_list.append(training_detailed_accuracies[i])
    for i in range(len(test_detailed_accuracies)):
        net.performance_list.append(test_detailed_accuracies[i])


def evaluate_classifier_dataset(net, dataset, verbose):
    accuracies = np.zeros([dataset.num_included_feature_types], float)
    costs = np.zeros([dataset.num_included_feature_types], float)
    detailed_accuracies = np.zeros([dataset.num_included_features], float)
    detailed_counts = np.zeros([dataset.num_included_features], float)

    for i in range(len(dataset.x)):
        o, h, o_cost = net.test_item(dataset.x[i], dataset.y[i])
        guess_list = []
        actual_list = []

        output_string1 = "\tOutputs |"
        output_string2 = "\tTargets |"
        output_string3 = "\tCost    |"
        output_string4 = "\tActual"
        output_string5 = "\tGuess"

        if verbose:
            print("Event {}".format(i))

        for j in range(dataset.num_included_feature_types):

            feature_type = dataset.included_feature_type_list[j]

            start_index = dataset.included_fv_indexes[j][0]
            stop_index = dataset.included_fv_indexes[j][1]

            current_costs = o_cost[start_index:stop_index+1]
            current_sum_cost = (current_costs ** 2).sum()
            costs[j] += current_sum_cost

            current_o = o[start_index:stop_index+1]
            current_y = dataset.y[i][start_index:stop_index+1]

            guess_index = np.argmax(current_o.detach().cpu().numpy())
            actual_index = np.argmax(current_y.detach().cpu().numpy())

            guess_score = current_o[guess_index]
            actual_score = current_y[actual_index]

            guess_label = dataset.feature_list_dict[feature_type][guess_index]
            actual_label = dataset.feature_list_dict[feature_type][actual_index]

            guess_list.append((guess_index, guess_label, guess_score))
            actual_list.append((actual_index, actual_label, actual_score))

            feature_index = dataset.included_feature_index_dict[actual_label]
            detailed_counts[feature_index] += 1

            if guess_index == actual_index:
                accuracies[j] += 1
                detailed_accuracies[feature_index] += 1

            for k in range(dataset.feature_type_size_dict[feature_type]):
                output_string1 += " {:0.2f}".format(current_o[k])
                output_string2 += " {:0.2f}".format(current_y[k])
                output_string3 += " {:0.2f}".format(current_costs[k])
            output_string1 += " |"
            output_string2 += " |"
            output_string3 += " |"
            output_string4 += "\t{:2s} {:10s} {:0.3f}".format(str(actual_list[j][0]), str(actual_list[j][1]), actual_list[j][2])
            output_string5 += "\t{:2s} {:10s} {:0.3f}".format(str(guess_list[j][0]), str(guess_list[j][1]), guess_list[j][2])

        if verbose:
            print(output_string1)
            print(output_string2)
            print(output_string3)
            print(output_string4)
            print(output_string5)
            print()

    costs /= len(dataset.x)
    for i in range(dataset.num_included_feature_types):
        feature_type = dataset.included_feature_type_list[i]
        costs[i] /= dataset.included_feature_type_size_dict[feature_type]
    accuracies = accuracies / len(dataset.x)

    for i in range(len(detailed_accuracies)):
        if detailed_counts[i] != 0:
            detailed_accuracies[i] = detailed_accuracies[i] / detailed_counts[i]
        else:
            detailed_accuracies[i] = np.nan

    return accuracies, costs, detailed_accuracies








