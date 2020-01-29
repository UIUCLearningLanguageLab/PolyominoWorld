import numpy as np
import sys
import torch
import time


def train_a(net, training_set, test_set, num_epochs, optimizer, learning_rate, shuffle_sequences, shuffle_events,
            output_freq, verbose):

    print("Training {}-{} on {} epochs".format(net.x_type, net.y_type, num_epochs))
    training_set.create_xy(net, False, False)
    test_set.create_xy(net, False, False)

    net.performance_list = [net.current_epoch, 0]
    if net.y_type == 'FeatureVector':
        evaluate_classifier(net, training_set, test_set, False)
    elif net.y_type == 'WorldState' and net.x_type == 'WorldState':
        evaluate_autoassociator(net, training_set, test_set, False)
    else:
        print("Unrecognized net type {} {}".format(net.x_type, net.y_type))
        sys.exit()

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

            if net.y_type == 'FeatureVector':
                evaluate_classifier(net, training_set, test_set, verbose)
            elif net.y_type == 'WorldState' and net.x_type == 'WorldState':
                evaluate_autoassociator(net, training_set, test_set, verbose)
            else:
                print("Unrecognized net type {} {}".format(net.x_type, net.y_type))
                sys.exit()

            net.save_network_properties()
            net.save_network_weights()
            net.save_network_performance()

    net.save_network_states(training_set)


def evaluate_autoassociator(net, training_set, test_set, verbose):
    training_cost = 0
    test_cost = 0

    for i in range(len(training_set.x)):
        o, h, o_cost = net.test_item(training_set.x[i], training_set.y[i])
        training_cost += o_cost.detach().cpu().numpy().sum()

        if verbose:
            output_string = "Training Event {}".format(i)
            for item in training_set.label_list[i]:
                output_string += " {:10s}".format(str(item))
            output_string += "{:0.1f}".format(o_cost.sum())
            print(output_string)

    for i in range(len(test_set.x)):
        o, h, o_cost = net.test_item(test_set.x[i], test_set.y[i])
        test_cost += o_cost.detach().cpu().numpy().sum()

        if verbose:
            output_string = "Test Event {}".format(i)
            for item in test_set.label_list[i]:
                output_string += " {:10s}".format(str(item))
            output_string += "{:0.1f}".format(o_cost.sum())
            print(output_string)

    training_cost = training_cost / len(training_set.x)
    test_cost = test_cost / len(test_set.x)
    print("Epoch:{}     training cost: {:0.4f}    test cost: {:0.4f}  took: {:0.2f}s".format(net.current_epoch,
                                                                                             training_cost, test_cost,
                                                                                             net.performance_list[1]))

    net.performance_list.append(training_cost)
    net.performance_list.append(test_cost)
    for i in range(training_set.num_included_feature_types):
        net.performance_list.append(np.nan)
        net.performance_list.append(np.nan)
        net.performance_list.append(np.nan)
        net.performance_list.append(np.nan)


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








