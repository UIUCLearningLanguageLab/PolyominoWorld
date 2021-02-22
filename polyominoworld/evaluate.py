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
                       cost_avg_train: Optional[float] = None,
                       cost_avg_valid: Optional[float] = None,
                       ):
    device = "gpu" if configs.Training.gpu else "cpu"
    output_string = f"Epoch:{epoch:>3} | "
    if cost_avg_train is not None and cost_avg_valid is not None:
        output_string += f"cost-train={cost_avg_train:0.2f} cost-valid={cost_avg_valid:0.2f} "
    else:
        output_string += 'Skipping evaluation'
    output_string += f"took:{cumulative_time:>6.0f}s on {device}"

    print(output_string, flush=True)


def evaluate_network(net: Network,
                     dataset: DataSet,
                     criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                     ) -> Dict[str, np.array]:

    if net.params.y_type == 'features':
        return evaluate_classifier(net, dataset, criterion_all)
    elif net.params.y_type == 'world':
        return evaluate_autoassociator(net, dataset, criterion_all)
    else:
        raise AttributeError("y_type not recognized")


def evaluate_classifier(net: Network,
                        dataset: DataSet,
                        criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                        ) -> Dict[str, float]:

    res = {}

    for event in dataset.generate_events():

        x = event.get_x(net.params.x_type)
        y = event.get_y(net.params.y_type)

        o = net.forward(x)
        costs_by_feature = criterion_all(o, y).detach().cpu().numpy()

        # collect cost for each feature
        for n, feature_label in enumerate(event.feature_vector.labels):
            performance_name = f'cost_{feature_label}-{dataset.name}'
            res.setdefault(performance_name, 0.0)
            res[performance_name] += costs_by_feature[n]

    # divide performance sum by count
    for performance_name in res:
        res[performance_name] /= len(dataset)

    # average cost
    res[f'cost_avg_{dataset.name}'] = np.mean([performance for name, performance in res.items()
                                               if name.startswith('cost')])

    return res


def evaluate_autoassociator(net: Network,
                            dataset: DataSet,
                            criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                            ) -> Dict[str, np.array]:

    raise NotImplementedError  # TODO ph february 21 2021


    # for each item in the dataset
    for event in dataset.generate_events():
        event: Event
        # get x and y
        x = event.get_x(self.params.x_type)
        y = event.get_y(self.params.y_type)

        # run the item through the network, getting cost and network activation values
        o = net.forward(x)
        costs_by_item = criterion_all(o, y)
        o_vector = o.detach().cpu().numpy()
        current_sum_cost = costs_by_item.sum()  # todo why sum?
        world_size = int(len(o_vector)/3)

        # use the output vector to calculate which cells are "on", and what color they are
        color_pred_list = []
        position_pred_list = []
        min_x = dataset.num_rows + 1
        min_y = dataset.num_cols + 1
        for j in range(world_size):
            pred_rgb = np.array([o_vector[j], o_vector[j + world_size], o_vector[j + world_size * 2]])
            distance_vector = np.linalg.norm(dataset.color_rgb_matrix - pred_rgb, axis=1)
            pred_index = np.argmin(distance_vector)
            pred_label = dataset.all_color_labels[pred_index]
            if pred_label != 'grey':
                # todo this may need to be num_columns, not num_rows, depending on how rXc matrix is flattened
                coordinates = [math.floor(j / dataset.num_rows), j % dataset.num_rows]
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
        for j in range(len(dataset.master_shape_label_list)):
            if dataset.master_shape_position_list[j] == position_pred_set:
                shape_pred_label = dataset.master_shape_label_list[j]
                break
        if shape_pred_label is None:
            shape_pred_label = random.choice(dataset.master_shapes)

        if len(color_pred_list) > 0:
            size_pred = len(color_pred_list)
        else:
            size_pred = random.choice([1, 2, 3, 4])

        if len(color_pred_list) == 0:
            color_pred_list = [random.choice(dataset.master_color_labels)]
        color_pred_label = random.choice(color_pred_list)

        # print()
        # print(i, label_list)
        # print(position_pred_set, shape_pred_label)
        # print(len(color_pred_list), size_pred)
        # print(color_pred_list, color_pred_label)

        for j in range(dataset.num_included_feature_types):
            feature_type = dataset.included_feature_types[j]
            true_label = label_list[j]
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

    return accuracies, costs
