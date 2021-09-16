"""

Plot one or more curves corresponding to a single performance (e.g. accuracy, error, etc.),
and compare performance across multiple parameter configurations.

Note:
In order to plot results,
we need to get results form the shared drive.
To look for results on the shared drive, we use ludwig.
We use params.param2requests to tell ludwig which jobs we would like results for.
To tell ludwig where to look for results,
create an environment variable "LUDWIG_MNT" that points to the path where ludwig_data is mounted on your machine.
"""

from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from polyominoworld.figs import plot_summary_fig, rank_label_for_legend_order, make_y_label
from polyominoworld.summary import make_summary
from polyominoworld.params import param2default, param2requests

# which results to plot
PERFORMANCE_NAME: str = 'acc_shape_test'  # name of performance curve to plot

# available performance names:
# {1}_{2}_{3}
# 1: cost, acc
# 2: shape, color, size (for cost and acc); shape-monomino, ..., color-red, ..., size-1, ..... (for cost only)
# 3: train, test
# 4 cumulative_seconds


# figure settings
LABELS: Optional[List[str]] = None  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 6)  # in inches
CONFIDENCE: float = 0.95
TITLE = ''

# for each job, save a summary, used for plotting
summaries = []
project_name = 'PolyominoWorld'
for param_path, label in gen_param_paths(project_name,
                                         param2requests,
                                         param2default,
                                         # isolated=True,
                                         # runs_path=Path(__file__).parent.parent / 'runs',
                                         ludwig_data_path=None,
                                         label_n=True):
    summary = make_summary(PERFORMANCE_NAME, param_path, label, CONFIDENCE)  # summary contains: x, mean_y, std_y, label, n
    summaries.append(summary)
    print(f'--------------------- End section {param_path.name}')
    print()

# sort data
summaries = sorted(summaries, key=lambda s: rank_label_for_legend_order(s[3]))
if not summaries:
    raise SystemExit('No data found')

# plot
plot_summary_fig(summaries,
                 x_label='Training Step',
                 y_label=make_y_label(PERFORMANCE_NAME),
                 title=TITLE,
                 figsize=FIG_SIZE,
                 y_lims=[0, 1] if 'acc' in PERFORMANCE_NAME else None,
                 legend_labels=LABELS,
                 legend_loc='best',
                 )
