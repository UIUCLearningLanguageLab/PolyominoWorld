import numpy as np
import random
import math
import torch
from typing import Union, Optional, Dict

from polyominoworld.network import Network
from polyominoworld.dataset import DataSet
from polyominoworld.helpers import Event
from polyominoworld import configs


def print_eval_summary(epoch: int,
                       cumulative_time: float,
                       cost_avg_train: float,
                       cost_avg_valid: float,
                       acc_avg_train: float,
                       acc_avg_valid: float,
                       ):
    device = "gpu" if configs.Training.gpu else "cpu"
    output_string = f"Epoch:{epoch:>3} | "
    output_string += f"cost-train={cost_avg_train:0.2f} cost-valid={cost_avg_valid:0.2f} | "
    output_string += f"acc-train={acc_avg_train:0.2f} acc-valid={acc_avg_valid:0.2f} | "
    output_string += f"took:{cumulative_time:>6.0f}s on {device}"

    print(output_string, flush=True)


def evaluate_network(net: Network,
                     dataset: DataSet,
                     criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                     ) -> Dict[str, np.array]:

    # TODO when training auto-associator, also train a classifier (with its hidden layer as input) alongside it,
    #  that can be evaluated on classification accuracy here

    # evaluate cost and accuracy classifying features in the world
    if net.params.y_type == 'features':
        return evaluate_classification(net, dataset, criterion_all)

    # evaluate cost of reconstructing the world vector given the world vector as input
    elif net.params.y_type == 'world':
        return evaluate_reconstruction(net, dataset, criterion_all)

    else:
        raise AttributeError("y_type not recognized")


def evaluate_classification(net: Network,
                            dataset: DataSet,
                            criterion_all: torch.nn.BCEWithLogitsLoss,
                            ) -> Dict[str, float]:
    """
    return cost and accuracy for each feature, averaged across all samples in provided dataset.
    """

    res = {}

    for event in dataset.get_events():

        x = event.get_x(net.params.x_type)
        y = event.get_y(net.params.y_type)

        o = net.forward(x)
        costs_by_feature = criterion_all(o, y).detach().cpu().numpy()
        o = o.detach().cpu().numpy()

        # collect cost for each feature
        for n, feature_label in enumerate(event.feature_vector.get_feature_labels()):
            performance_name = f'cost_{feature_label}-{dataset.name}'
            res.setdefault(performance_name, 0.0)
            res[performance_name] += costs_by_feature[n]

        # collect accuracy for each feature_type
        for feature_type, o_ids in event.feature_vector.feature_type2ids.items():
            performance_name = f'acc_{feature_type}-{dataset.name}'
            res.setdefault(performance_name, 0.0)
            o_restricted = o[o_ids]  # logits restricted to one feature_type
            y_restricted = event.get_y(net.params.y_type)[o_ids]
            if np.argmax(o_restricted) == np.argmax(y_restricted):
                res[performance_name] += 1

    # divide performance sum by count
    num_events = len(dataset)
    for performance_name in res:
        res[performance_name] /= num_events

    # add average cost and accuracy
    res[f'cost_avg_{dataset.name}'] = np.mean([performance for name, performance in res.items()
                                               if name.startswith('cost')])
    res[f'acc_avg_{dataset.name}'] = np.mean([performance for name, performance in res.items()
                                              if name.startswith('acc')])

    if configs.Evaluation.means_only:
        res = {k: v for k, v in res.items() if 'avg' in k}

    return res


def evaluate_reconstruction(net: Network,
                            dataset: DataSet,
                            criterion_all: torch.nn.MSELoss,
                            ) -> Dict[str, float]:
    """
    return cost reconstructing the world from world input, e.g. "reconstruction cost".
    """

    performance_name = f'cost-reconstruction_avg_{dataset.name}'

    res = {performance_name: 0}

    for event in dataset.get_events():
        x = event.get_x(net.params.x_type)
        y = event.get_y(net.params.y_type)

        o = net.forward(x)
        costs = criterion_all(o, y).detach().cpu().numpy()
        res[performance_name] += np.mean(costs)

    # divide sum by count
    res[performance_name] /= len(dataset)

    return res


def evaluate_autoassociator(net: Network,
                            dataset: DataSet,
                            criterion_all: torch.nn.MSELoss,
                            ) -> Dict[str, float]:

    # TODO ph february 21 2021: this function evaluates auto-associator classification of features.
    #  but it is probably better to evaluate the auto-associator hidden layer via a classifier.
    #  this would spare us the need to come up with code for evaluating classification accuracy,
    #  because we would just use existing code

    color_rgb_matrix = np.zeros([len(master_color_labels), 3], float)
    for n, color in enumerate(master_color_labels):
        rgb = configs.World.color2rgb[color]
        color_rgb_matrix[n] = rgb

    # for each item in the dataset
    for event in dataset.get_events():
        event: Event

        x = event.get_x(net.params.x_type)
        y = event.get_y(net.params.y_type)

        o = net.forward(x).detach().numpy()

        # use the output vector to calculate which cells are "on", and what color they are
        color_pred_list = []
        position_pred_list = []
        min_x = dataset.max_y + 1
        min_y = dataset.max_x + 1
        for j in range(world_size):
            pred_rgb = np.array([o[j], o[j + world_size], o[j + world_size * 2]])
            distance_vector = np.linalg.norm(color_rgb_matrix - pred_rgb, axis=1)
            pred_index = np.argmin(distance_vector)
            pred_label = all_color_labels[pred_index]
            if pred_label != 'grey':
                # todo this may need to be num_columns, not num_rows, depending on how rXc matrix is flattened
                coordinates = [math.floor(j / configs.World.max_x), j % configs.World.max_y]
                if coordinates[0] < min_x:
                    min_x = coordinates[0]
                if coordinates[1] < min_y:
                    min_y = coordinates[1]
                color_pred_list.append(pred_label)
                position_pred_list.append(coordinates)

        # convert the list of positions of cells that are "on", to a set of top-left-aligned active coordinates
        for j in range(len(position_pred_list)):
            position_pred_list[j] = [position_pred_list[j][0]-min_x, position_pred_list[j][1] - min_y]
        position_pred_set = set(tuple(tuple(x) for x in position_pred_list))

        # check to see if the set of top-lef- aligned active coordinates matches the definition of any shapes
        shape_pred_label = None
        for j in range(len(master_shape_label_list)):
            if master_shape_position_list[j] == position_pred_set:
                shape_pred_label = master_shape_label_list[j]
                break
        if shape_pred_label is None:
            shape_pred_label = random.choice(master_shapes)

        if len(color_pred_list) > 0:
            size_pred = len(color_pred_list)
        else:
            size_pred = random.choice([1, 2, 3, 4])

        if len(color_pred_list) == 0:
            color_pred_list = [random.choice(master_color_labels)]
        color_pred_label = random.choice(color_pred_list)

        for feature_type in configs.World.feature_type2values:
            feature_index = dataset.included_feature_index_dict[true_label]
            costs[j] += current_sum_cost

            if feature_type == 'size':
                pred_label = size_pred
            elif feature_type == 'shape':
                pred_label = shape_pred_label
            elif feature_type == 'color':
                pred_label = color_pred_label
            elif feature_type == 'action':
                pred_label = None
            else:
                raise TypeError("Feature Type Not recognized While Evaluating Autoassociator")

            if true_label == pred_label:
                accuracies[j] += 1

    costs /= len(dataset.x)
    accuracies = accuracies / len(dataset.x)

    return res
