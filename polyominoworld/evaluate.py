import numpy as np
import torch
from typing import Union, Dict, Tuple, Optional, List
from numpy.linalg import pinv

from polyominoworld.network import Network
from polyominoworld.dataset import DataSet

from polyominoworld import configs


def print_eval_summary(epoch: int,
                       cumulative_time: float,
                       cost_avg_train: float,
                       cost_avg_valid: float,
                       acc_avg_train: float,
                       acc_avg_valid: float,
                       ):
    device = "gpu" if configs.Training.gpu else "cpu"
    output_string = f"Epoch={epoch:04} | "
    output_string += f"cost-train={cost_avg_train:0.2f} cost-valid={cost_avg_valid:0.2f} | "
    output_string += f"acc-train={acc_avg_train:0.2f} acc-valid={acc_avg_valid:0.2f} | "
    output_string += f"minutes elapsed={int(cumulative_time/60):03}min with device={device}"

    print(output_string, flush=True)


def evaluate_network(net: Network,
                     dataset: DataSet,
                     criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                     ) -> Dict[str, np.array]:

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
            performance_name = f'cost_{feature_label}_{dataset.name}'
            res.setdefault(performance_name, 0.0)
            res[performance_name] += costs_by_feature[n]

        # collect accuracy for each feature_type
        for feature_type, o_ids in event.feature_vector.feature_type2ids.items():
            performance_name = f'acc_{feature_type}_{dataset.name}'
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

    # TODO evaluate the auto-associator hidden layer using a linear readout (and moore penrose pseudo inverse).

    raise NotImplementedError


def make_l_and_p(data: DataSet,
                 net: Network,
                 feature_type: str,
                 h_ids: Optional[List[int]] = None,
                 non_linearity: bool = True,
                 state_is_random: bool = False,
                 state_is_input: bool = False,
                 ) -> Tuple[np.array, np.array]:
    """
    build data structures for evaluating linear readout
    """

    lis = []
    pis = []
    for event in data.get_events():

        # make l
        if feature_type == 'shape':
            li = [1 if event.shape == s else 0 for s in configs.World.master_shapes]
        elif feature_type == 'color':
            li = [1 if event.color == s else 0 for s in configs.World.master_colors]
        elif feature_type == 'size':
            li = [1 if event.size == s else 0 for s in configs.World.master_sizes]
        else:
            raise AttributeError('Invalid feature type')
        lis.append(li)

        # make p
        x = event.world_vector.vector

        if state_is_input:
            state = x.numpy()
        else:
            if h_ids is None:
                z_h = net.h_x.weight @ x + net.h_x.bias  # don't forget the bias!
                z_h = z_h.numpy()
            else:
                term1 = net.h_x.weight.numpy()[h_ids, :] @ x.numpy()
                term2 = net.h_x.bias.numpy()[h_ids]
                z_h = term1 + term2

            if non_linearity:
                state = np.tanh(z_h)
            else:
                state = z_h

        if state_is_random:
            state = np.random.permutation(state)

        pi = state.T
        pis.append(pi)

    l = np.array(lis).T  # l has columns of localist target vectors
    p = np.array(pis).T  # p has columns of representation vectors

    return l, p


def calc_accuracy(l_matrix_correct: np.array,
                  l_matrix_predicted: np.array,
                  ) -> float:
    """
    calculate accuracy of predicted class labels against target labels.
    used for evaluating linear readout.
    """
    num_correct = 0
    for li_correct, li_predicted in zip(l_matrix_correct.T, l_matrix_predicted.T):
        assert np.sum(li_correct) == 1.0
        id_correct = np.argmax(li_correct)
        id_predicted = np.argmax(li_predicted)
        if id_correct == id_predicted:
            num_correct += 1
    return num_correct / len(l_matrix_correct.T)


def evaluate_linear_readout(data: DataSet,
                            net: Network,
                            feature_type: str,
                            h_ids: Optional[List[int]] = None,
                            non_linearity: bool = True,
                            **kwargs,
                            ) -> float:
    """
    compute accuracy of linear readout.
    """

    # make l and P
    l, p = make_l_and_p(data, net, feature_type, h_ids, non_linearity, **kwargs)
    # compute W
    w = l @ pinv(p)
    # compute linear readout l
    l_predicted = w @ p  # [num features, num instances]
    # compute accuracy
    res = calc_accuracy(l, l_predicted)

    return res