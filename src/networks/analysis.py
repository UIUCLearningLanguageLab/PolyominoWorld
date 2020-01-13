import numpy as np
import time
import sys
import torch


def train_a(net, training_set, test_set,
            num_epochs, learning_rate, shuffle_sequences, shuffle_events,
            output_freq, verbose):

    if net.net_type == 'classifier':
        x_type = 'world_state'
        y_type = 'feature_vector'
    elif net.net_type == 'autoassociator':
        x_type = 'world_state'
        y_type = 'world_state'
    else:
        print("Unrecognized net type {}".format(net.net_type))
        sys.exit()

    print("Training {} epochs on {}-{}-{} {} on dataset {}, testing on dataset {}".format(num_epochs,
                                                                                          net.input_size,
                                                                                          net.hidden_size,
                                                                                          net.output_size,
                                                                                          net.net_type,
                                                                                          training_set.name,
                                                                                          test_set.name))

    start_time = time.time()
    for i in range(num_epochs):
        training_set.create_xy(x_type, y_type, shuffle_sequences, shuffle_events)
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

        for j in range(len(training_set.x)):
            net.train_item(training_set.x[j], training_set.y[j], optimizer)

        if i % output_freq == 0:

            took = time.time() - start_time
            start_time = time.time()

            if net.net_type == 'classifier':
                training_costs, training_pc = evaluate_classifier(net, training_set, verbose)
                test_costs, test_pc = evaluate_classifier(net, test_set, verbose)

                output_string = "Epoch:{} {:0.2f}s |".format(i, took)
                for j in range(training_set.num_included_feature_types):
                    output_string += "  {:0.2f}-{:0.2f}".format(training_costs[j], test_costs[j])
                output_string += "  | "
                for j in range(training_set.num_included_feature_types):
                    output_string += "  {:0.2f}-{:0.2f}".format(training_pc[j], test_pc[j])
                print(output_string)

            elif net.net_type == 'autoassociator':
                cost = 0
                for j in range(len(training_set.x)):
                    o, h, o_cost = net.test_item(training_set.x[j], training_set.y[j])
                    cost += o_cost.sum()
                print("Epoch:{} {:16s} costs: {:0.1f}  took:{:0.2f}s.".format(i, training_set.name, cost, took))

    print("Training complete. Saving hidden states and weights.")
    for i in range(len(training_set.x)):
        if net.net_type == 'classifier':
            out, h, o_cost = net.test_item(training_set.x[i], training_set.y[i])
        elif net.net_type == 'autoassociator':
            out, h, o_cost = net.test_item(training_set.x[i], training_set.x[i])
        else:
            print("Network type not found")
            sys.exit(2)

        net.save_network_state(training_set.x[i], training_set.y[i], out, h)

    net.generate_states_file()
    net.generate_weights_file()


def evaluate_classifier(net, the_dataset, verbose):
    the_dataset.create_xy('world_state', 'feature_vector', False, False)

    accuracy_array = np.zeros([the_dataset.num_included_feature_types], float)
    costs = np.zeros([the_dataset.num_included_feature_types], float)

    for i in range(len(the_dataset.x)):
        o, h, o_cost = net.test_item(the_dataset.x[i], the_dataset.y[i])
        actual_score_list = []

        for j in range(the_dataset.num_included_feature_types):

            start_index = the_dataset.included_fv_indexes[j][0]
            stop_index = the_dataset.included_fv_indexes[j][1]

            current_cost = (o_cost[start_index:stop_index] ** 2).sum()
            costs[j] += current_cost

            current_o = o[start_index:stop_index]
            current_y = the_dataset.y[i][start_index:stop_index]
            current_guess_index = np.argmax(current_o.detach().numpy())
            current_actual_index = np.argmax(current_y.detach().numpy())

            actual_score = current_o[current_actual_index]

            actual_score_list.append(actual_score)
            if current_guess_index == current_actual_index:
                accuracy_array[j] += 1

        if verbose:
            output_string = "{}  {}".format(i, the_dataset.label_list[i])
            for j in range(the_dataset.num_included_features):
                output_string += "  {:0.3f}".format(actual_score_list[j])
            print(output_string)

            for j in range(the_dataset.num_included_feature_types):
                feature_type = the_dataset.included_feature_type_list[j]
                start_index = the_dataset.included_fv_indexes[j][0]
                stop_index = the_dataset.included_fv_indexes[j][1]
                current_o = o[start_index:stop_index]
                current_y = the_dataset.y[start_index:stop_index]
                print("\t\t{}} Actual:  {}".format(feature_type, current_y))
                print("\t\t{}} Guess:   {}".format(feature_type, current_o))
            print()

    costs /= len(the_dataset.x)
    for i in range(the_dataset.num_included_feature_types):
        feature_type = the_dataset.included_feature_type_list[i]
        costs[i] /= the_dataset.included_feature_type_size_dict[feature_type]
    pc = accuracy_array / len(the_dataset.x)

    return costs, pc
