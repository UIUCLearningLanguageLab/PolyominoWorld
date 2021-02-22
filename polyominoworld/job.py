"""
This module is used by Ludwig to run a single job.
A job consists of :
- creating a world and creating a dataset from it
- training a network on the data
- evaluating a network periodically during training
- saving evaluation data to the shared drive used by Ludwig
"""

import pandas as pd
from pathlib import Path
import torch
import time
from typing import Dict, List

from polyominoworld.dataset import DataSet
from polyominoworld.world import World
from polyominoworld.network import Network
from polyominoworld.params import Params
from polyominoworld import configs
from polyominoworld.evaluate import evaluate_network, print_eval_summary


def main(param2val):
    """run 1 job, defined by 1 hyper-parameter configuration in "param2val" a dict created by Ludwig"""

    params = Params.from_param2val(param2val)
    print(params, flush=True)

    save_path = Path(param2val['save_path'])  # use this path to save any files
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # generate world + load into dataset
    world_train = World(params)
    world_valid = World(params)  # for validation/testing
    data_train = DataSet(world_train.generate_sequences(), params, 'train')
    data_valid = DataSet(world_valid.generate_sequences(), params, 'valid')

    # network
    net = Network(params)
    if configs.Training.gpu:
        if torch.cuda.is_available():
            net.cuda()
        else:
            raise RuntimeError('CUDA is not available')

    # loss function
    if params.criterion == 'mse':
        if params.y_type == 'world':  # use MSE only with auto-associator
            raise RuntimeError('MSE loss should only be used with auto-associator')
        else:
            criterion_avg = torch.nn.MSELoss()
            criterion_all = torch.nn.MSELoss(reduction='none')  # for evaluation
    elif params.criterion == 'bce':
        criterion_avg = torch.nn.BCEWithLogitsLoss()  # sigmoid function and then a binary cross entropy
        criterion_all = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        raise AttributeError(f'Invalid arg to criterion')

    # optimizer
    if params.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate)
    elif params.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=params.learning_rate)
    else:
        raise AttributeError(f'Invalid arg to optimizer')

    # train loop  # todo eval before training
    print("Starting training {}-to-{} model on {} epochs".format(params.x_type, params.y_type, params.num_epochs))
    start_time = time.time()
    performance_curves_dict = {}  # maps name of a performance curve (e.g. "loss") to a tuple (epoch, performance_value)
    for epoch in range(params.num_epochs):

        # train
        net.train()
        for event in data_train.generate_samples():

            x = event.get_x(net.params.x_type)
            y = event.get_y(net.params.y_type)

            o = net.forward(x)
            loss = criterion_avg(o, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval
        net.eval()

        if epoch % configs.Evaluation.epoch_interval == 0:
            # collect time data
            cumulative_time = time.time() - start_time
            performance_curves_dict.setdefault('seconds', []).append((epoch, cumulative_time))

            # compute performance data
            performances: Dict[str, float] = {}  # store both train and valid data
            if not configs.Evaluation.skip:

                # for train and valid data
                for data in [data_train, data_valid]:
                    # compute performance data
                    performances_ = evaluate_network(net, data, criterion_all)
                    performances.update(performances_)

                    # collect performance data for plotting with Ludwig-Viz
                    for name, val in performances_.items():
                        performance_curves_dict.setdefault(name, []).append((epoch, val))

            # print
            print_eval_summary(epoch,
                               cumulative_time,
                               performances['cost_avg_train'],
                               performances['cost_avg_valid'],
                               )

    # prepare collected data for returning to Ludwig (which saves data to shared drive)
    performance_curves = []
    for name, curves in performance_curves_dict.items():
        print(f'Making pandas series with name={name} and length={len(curves)}', flush=True)
        index, curve = zip(*curves)
        s = pd.Series(curve, index=index)
        s.name = name
        performance_curves.append(s)

    print('Done with job.main', flush=True)

    return performance_curves
