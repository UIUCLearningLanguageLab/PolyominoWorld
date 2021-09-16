"""
Plot one or more curves corresponding, each corresponding to a different performance (e.g. accuracy, error, etc.),
for a single parameter configuration.
To compare performance on a single performance across parameter configurations, see plot_performance_comparison.py

Note:
In order to plot results,
we need to get results form the shared drive.
To look for results on the shared drive, we use ludwig.
We use params.param2requests to tell ludwig which jobs we would like results for.
To tell ludwig where to look for results,
create an environment variable "LUDWIG_MNT" that points to the path where ludwig_data is mounted on your machine.
"""

from typing import Optional, List, Tuple
import yaml
from pathlib import Path

from ludwig.results import gen_param_paths

from polyominoworld.figs import plot_summary_fig, rank_label_for_legend_order, make_y_label
from polyominoworld.summary import make_summary
from polyominoworld.utils import is_leftout
from polyominoworld.params import param2default, param2requests, Params

# names of performance curves to plot
PERFORMANCE_NAMES: List[str] = ['cost_color-red_test']

# available PERFORMANCE_NAMES:
# {1}_{2}_{3}
# 1: cost, acc
# 2: shape, color, size (for cost and acc); shape-monomino, ..., color-red, ..., size-1, ..... (for cost only)
# 3: train, test
# 4 cumulative_seconds


# figure settings
LABELS: Optional[List[str]] = None  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (8, 6)  # in inches
CONFIDENCE: float = 0.95

# check that patterns contains performance curves of same type (e.g. cost, acc)
y_labels = []
for pn in PERFORMANCE_NAMES:
    y_labels.append(pn.split('_')[0])
if len(set(y_labels)) > 1:
    raise ValueError(f'Found performance name belonging to different types of performances {y_labels} '
                     f'but can use one type only.')
else:
    y_label = {'acc': 'Accuracy', 'cost': 'Error'}[y_labels[0]]

# for each job, save a summary, used for plotting
title = None
project_name = 'PolyominoWorld'
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         # isolated=True,
                                         # runs_path=Path(__file__).parent.parent / 'runs',
                                         ludwig_data_path=None,
                                         label_n=True):

    summaries = []

    # load params to get info about what was leftout from data
    with (param_path / 'param2val.yaml').open('r') as f:
        param2val = yaml.load(f, Loader=yaml.FullLoader)
    params = Params.from_param2val(param2val)

    title = label
    for pn in PERFORMANCE_NAMES:

        # check that we are not trying to plot something that was leftout
        performance_type, feature_label, data_name = pn.split('_')
        if performance_type == 'cost':
            feature_type, feature_value = feature_label.split('-')
            if data_name == 'train':
                leftout_colors_and_shapes = params.train_leftout_colors + params.train_leftout_shapes
            elif data_name == 'test':
                leftout_colors_and_shapes = params.test_leftout_colors + params.test_leftout_shapes
            else:
                raise AttributeError('Invalid arg to data.name')
            if is_leftout(feature_value, leftout_colors_and_shapes):
                raise RuntimeError(f'Feature "{feature_value}" was leftout from {data_name} data.')

        # make summary
        label = make_y_label(pn.replace('cost_', '').replace('acc_', ''))
        summary = make_summary(pn, param_path, label, CONFIDENCE)  # summary contains: x, mean_y, std_y, label, n
        summaries.append(summary)
    print(f'--------------------- End section {param_path.name}')
    print()

    # sort data
    summaries = sorted(summaries, key=lambda s: rank_label_for_legend_order(s[3]))
    if not summaries:
        raise SystemExit('No data found')

    # plot
    fig = plot_summary_fig(summaries,
                           x_label='Training Step',
                           y_label=y_label,
                           title=title,
                           figsize=FIG_SIZE,
                           y_lims=[0, 1] if 'Accuracy' in y_label else None,
                           legend_labels=LABELS,
                           legend_ncol=1,
                           )
