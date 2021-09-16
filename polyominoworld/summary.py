from pathlib import Path
import numpy as np
from typing import Tuple
import pandas as pd
from scipy.stats import sem, t
import yaml
import re

from polyominoworld.params import Params, param2default
from polyominoworld import configs


def make_summary(pattern: str,
                 param_path: Path,
                 label: str,
                 confidence: float,
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """
    load all csv files matching pattern and return mean and std across their contents
    """
    pattern = f'{pattern}.csv'
    series_list = [pd.read_csv(p, index_col=0, squeeze=True)
                   for p in param_path.rglob(pattern)]
    n = len(series_list)
    if not series_list:
        raise RuntimeError(f'Did not find any csv files matching pattern="{pattern}"')
    concatenated_df = pd.concat(series_list, axis=1)
    x = concatenated_df.index.values
    y_mean = concatenated_df.mean(axis=1).values.flatten()
    y_sem = sem(concatenated_df.values, axis=1)
    h = y_sem * t.ppf((1 + confidence) / 2, n - 1)  # margin of error

    # load params to get more info about pre-training
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

    # make legend label more informative by adding info about pre-training configuration
    if params.load_from_checkpoint != 'none':
        # load params of model that was loaded
        with (param_path.parent / params.load_from_checkpoint / 'param2val.yaml').open('r') as f:
            param2val_pre_training = yaml.load(f, Loader=yaml.FullLoader)
        pretrain_info = ''
        for k, v in param2default.items():
            if v != param2val_pre_training[k]:
                pretrain_info += f'{k}={param2val_pre_training[k]} (during pre-training)'
        label = re.sub(r'load_from_checkpoint=param_\d\d\d', pretrain_info, label, count=0, flags=0)
    else:
        label = label.replace('load_from_checkpoint=none', configs.Figs.NO_PRE_TRAINING_STRING)

    return x, y_mean, h, label, n
