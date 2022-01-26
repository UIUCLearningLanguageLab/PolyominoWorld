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
from typing import Dict, List, Tuple, Union
import random
import numpy as np
from itertools import count

from polyominoworld.dataset import DataSet
from polyominoworld.utils import get_leftout_positions, get_test_data_kwargs
from polyominoworld.world import World
from polyominoworld.network import Network
from polyominoworld.params import Params
from polyominoworld import configs
from polyominoworld.evaluate import evaluate_network, print_eval_summary


def main(param2val):
    """run 1 job, defined by 1 hyper-parameter configuration in "param2val" a dict created by Ludwig"""

    params: Params = Params.from_param2val(param2val)
    print(params, flush=True)

    if params.y_type not in configs.ArgCheck.y_type:
        raise AttributeError('Invalid arg to y_type')

    if params.x_type not in configs.ArgCheck.x_type:
        raise AttributeError('Invalid arg to x_type')

    if params.batch_size < 4096:
        configs.Device.gpu = False
        print('Setting gpu=False')

    save_path = Path(param2val['save_path'])  # use this path to save any files
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # make world
    world = World(params)

    # make train dataset
    data_train = DataSet(world.generate_sequences(leftout_colors=params.train_leftout_colors,
                                                  leftout_shapes=params.train_leftout_shapes,
                                                  leftout_variants=params.train_leftout_variants,
                                                  leftout_positions=get_leftout_positions(params.train_leftout_half),
                                                  ),
                         seed=params.seed,
                         shuffle_events=params.shuffle_events,
                         shuffle_sequences=params.shuffle_sequences,
                         name='train')

    # make test dataset based on (ordered by priority):
    # 1) custom instructions, or
    # 2) what is leftout from training data, or
    # 3) what is leftout from pretraining dataset, or
    # else: nothing is leftout
    test_data_kwargs = get_test_data_kwargs(param2val)
    data_test = DataSet(world.generate_sequences(**test_data_kwargs),
                        seed=params.seed,
                        shuffle_events=params.shuffle_events,
                        shuffle_sequences=params.shuffle_sequences,
                        name='test')

    assert data_train.sequences
    assert data_test.sequences

    # network
    net = Network(params)
    if params.load_from_checkpoint.startswith('param'):  # load weights from previous checkpoint
        path_tmp = Path(param2val['project_path']) / 'runs' / params.load_from_checkpoint
        print(f'Trying to load model from {path_tmp}')
        model_files = list(path_tmp.rglob('**/saves/model.pt'))
        print(f'Found {len(model_files)} saved models')
        path_cpt = random.choice(model_files)
        state_dict = torch.load(path_cpt)
        net.load_state_dict(state_dict)
        print(f'Loaded model from {path_cpt}')
    if configs.Device.gpu:
        if torch.cuda.is_available():
            net.cuda()
        else:
            raise RuntimeError('CUDA is not available')

    # loss function
    if params.criterion == 'mse':
        if params.y_type != 'world':  # use MSE only with auto-associator
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
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=params.learning_rates[0],
                                     )
    elif params.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=params.learning_rates[0],
                                    momentum=params.momenta[0],
                                    nesterov=params.nesterov,
                                    weight_decay=params.weight_decay,
                                    )
    else:
        raise AttributeError(f'Invalid arg to optimizer')

    # init
    print("Starting training {}-to-{} model on {:,} steps".format(params.x_type, params.y_type, params.num_steps))
    start_time = time.time()
    step = 0
    epoch = 0
    lr = 0
    is_finished_training = False
    performance_data: Dict[str, List[Tuple[int, float]]] = {}

    # eval before training
    evaluate_on_train_and_test(criterion_all,
                               data_train,
                               data_test,
                               step,
                               net,
                               performance_data,
                               start_time)

    print_eval_summary(epoch,
                       step,
                       params.num_steps,
                       lr,
                       performance_data,
                       net.params.y_type,
                       )

    # save network weights for visualizing later
    #torch.save(net.state_dict(), save_path / f'model_{step:012}.pt')
    #torch.save(net.state_dict(), save_path / 'model.pt')

    # compute learning rate schedule (linear increase, then linear decrease)
    lr1, lr2, lr3 = params.learning_rates
    lrs_inc = list(np.linspace(lr1, lr2, num=params.num_steps // 2 + 1))
    lrs_dec = list(np.linspace(lr2, lr3, num=params.num_steps // 2 + 1))
    learning_rates = (lr for lr in lrs_inc + lrs_dec)

    # compute momentum schedule (linear decrease, then linear increase)
    m1, m2, m3 = params.momenta
    ma_dec = list(np.linspace(m1, m2, num=params.num_steps // 2 + 1))
    ma_inc = list(np.linspace(m2, m3, num=params.num_steps // 2 + 1))
    momenta = (m for m in ma_dec + ma_inc)

    # precompute training tensors
    events = data_train.get_events()
    xs = torch.stack([event.get_x(net.params.x_type) for event in events])
    ys = torch.stack([event.get_y(net.params.y_type) for event in events])

    # train loop
    for epoch in count(start=1, step=1):  # infinite counter

        # shuffle world_vector tensor at every epoch (to prevent over-fitting on a specific permutation)
        # careful: train evaluation data uses different data than what is trained on (due to different permutation)
        if params.shuffle_input:
            events = data_train.get_events()
            xs = torch.stack([event.get_x(net.params.x_type) for event in events])
            ys = torch.stack([event.get_y(net.params.y_type) for event in events])

        # shuffle data at start of epoch
        rand_ids = torch.randperm(len(xs))
        xs = xs[rand_ids]
        ys = ys[rand_ids]

        for x, y in zip(torch.split(xs, params.batch_size), torch.split(ys, params.batch_size)):
            step += 1

            # start with small learning rate and ramp up, then down again (super-convergence)
            lr = next(learning_rates)
            momentum = next(momenta)
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[0]['momentum'] = momentum

            # train
            net.train()
            o = net.forward(x)
            loss = criterion_avg(o, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # eval
            net.eval()
            if step % configs.Evaluation.step_interval == 0:
                evaluate_on_train_and_test(criterion_all,
                                           data_train,
                                           data_test,
                                           step,
                                           net,
                                           performance_data,
                                           start_time)

                # print
                print_eval_summary(epoch,
                                   step,
                                   params.num_steps,
                                   lr,
                                   performance_data,
                                   net.params.y_type,
                                   )

                # save network weights for visualizing later
                torch.save(net.state_dict(), save_path / f'model_{step:012}.pt')
                torch.save(net.state_dict(), save_path / 'model.pt')

            if step >= params.num_steps:
                is_finished_training = True
                break

        if is_finished_training:
            break

    # prepare collected data for returning to Ludwig (which saves data to shared drive)
    res: List[pd.Series] = []
    for name, curves in performance_data.items():
        index, curve = zip(*curves)
        s = pd.Series(curve, index=index)
        s.name = name
        res.append(s)

    print('Done with job.main', flush=True)

    return res


def evaluate_on_train_and_test(criterion_all: Union[torch.nn.BCEWithLogitsLoss, torch.nn.MSELoss],
                               data_train: DataSet,
                               data_test: DataSet,
                               step: int,
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
    performance_data.setdefault('cumulative_seconds', []).append((step, cumulative_time))

    # for train and test data
    for data in [data_train, data_test]:

        # skip evaluation on test data (useful only when debugging)
        if data.name == 'test' and configs.Evaluation.skip_test_data:
            continue

        # compute and collect performance data for plotting with Ludwig-Viz
        for name, val in evaluate_network(net, data, criterion_all).items():
            performance_data.setdefault(name, []).append((step, val))
