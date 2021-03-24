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
import numpy as np
from typing import Dict, List, Tuple, Union
import random

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

    # make world
    world = World(params)

    # make train dataset
    data_train = DataSet(world.generate_sequences(leftout_colors=params.leftout_colors,
                                                  leftout_shapes=params.leftout_shapes),
                         params,
                         'train')

    # make test/valid dataset
    if params.leftout_colors:
        leftout_colors_inverse = tuple([c for c in configs.World.master_colors if c not in params.leftout_colors])
    else:
        leftout_colors_inverse = ()
    if params.leftout_shapes:
        leftout_shapes_inverse = tuple([c for c in configs.World.master_shapes if c not in params.leftout_shapes])
    else:
        leftout_shapes_inverse = ()
    data_valid = DataSet(world.generate_sequences(leftout_colors=leftout_colors_inverse,
                                                  leftout_shapes=leftout_shapes_inverse),
                         params,
                         'valid')

    assert data_train.sequences
    assert data_valid.sequences

    # network
    net = Network(params)
    if params.load_from_checkpoint.startswith('param'):  # load weights from previous checkpoint
        path_tmp = Path(param2val['project_path']) / 'runs' / params.load_from_checkpoint
        path_cpt = random.choice(list(path_tmp.rglob('**/saves/model.pt')))
        state_dict = torch.load(path_cpt)
        net.load_state_dict(state_dict)
        print(f'Loaded weights from {path_cpt}')
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

    # init
    print("Starting training {}-to-{} model on {} epochs".format(params.x_type, params.y_type, params.num_epochs))
    start_time = time.time()
    epoch = 0
    performance_data: Dict[str, List[Tuple[int, float]]] = {}

    # eval before training
    evaluate_on_train_and_valid(criterion_all,
                                data_train,
                                data_valid,
                                epoch,
                                net,
                                performance_data,
                                start_time)

    # train loop
    for epoch in range(1, params.num_epochs + 1):  # start at 1 because evaluation at epoch=0 happens before training

        # train
        net.train()
        for event in data_train.get_events():

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
            evaluate_on_train_and_valid(criterion_all,
                                        data_train,
                                        data_valid,
                                        epoch,
                                        net,
                                        performance_data,
                                        start_time)

    # save network weights for visualizing later
    torch.save(net.state_dict(), save_path / 'model.pt')

    # prepare collected data for returning to Ludwig (which saves data to shared drive)
    res: List[pd.Series] = []
    for name, curves in performance_data.items():
        print(f'Making pandas series with name={name} and length={len(curves)}', flush=True)
        index, curve = zip(*curves)
        s = pd.Series(curve, index=index)
        s.name = name
        res.append(s)

    print('Done with job.main', flush=True)

    return res


def evaluate_on_train_and_valid(criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                                data_train: DataSet,
                                data_valid: DataSet,
                                epoch: int,
                                net: Network,
                                performance_data: Dict[str, List[Tuple[int, float]]],
                                start_time_train: time.time,
                                ) -> None:
    """
    save performance data to performance_data which will be saved to the shared drive by Ludwig.
    """

    # collect time data

    start_time_eval = time.time()
    cumulative_time = start_time_eval - start_time_train
    performance_data.setdefault('cumulative_seconds', []).append((epoch, cumulative_time))

    # for train and valid data
    for data in [data_train, data_valid]:

        # compute and collect performance data for plotting with Ludwig-Viz
        for name, val in evaluate_network(net, data, criterion_all).items():
            performance_data.setdefault(name, []).append((epoch, val))

    # print
    print_eval_summary(epoch,
                       performance_data['cumulative_seconds'][-1][1],
                       performance_data['cost_avg_train'][-1][1],
                       performance_data['cost_avg_valid'][-1][1],
                       performance_data['acc_avg_train'][-1][1],
                       performance_data['acc_avg_valid'][-1][1],
                       )
    print(f'Evaluation took {time.time() - start_time_eval} seconds')
