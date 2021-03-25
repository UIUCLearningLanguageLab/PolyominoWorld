"""
In order to plot results,
we need to get results form the shared drive.
To look for results on the shared drive, we use ludwig.
We use params.param2requests to tell ludwig which jobs we would like results for.
To tell ludwig where to look for results,
create an environment variable "LUDWIG_MNT" that points to the path where ludwig_data is mounted on your machine.
"""

from typing import Optional, List, Tuple

from ludwig.results import gen_param_paths

from polyominoworld.figs import make_summary_fig
from polyominoworld.summary import make_summary
from polyominoworld.params import param2default, param2requests

# which results to plot
PATTERN: str = 'acc_avg_train'  # name of performance curve to plot

# available patterns:
# {1}_{2}_{3}
# 1: cost, acc
# 2: shape, color, size, monomino, ..., red, ..., 1, .....
# 3: train, valid


# figure settings
LABELS: Optional[List[str]] = None  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: Optional[List[float]] = [0, 1.]
CONFIDENCE: float = 0.95
TITLE = ''


# param2requests

# 8, 16, 32, 64
# 0.01, 0.10, 0.20, 0.30, 0.40
#0.000001, 0.00001, 0.001, 0.01

# for each job, save a summary, used for plotting
summaries = []
project_name = 'PolyominoWorld'
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=None,
                                ludwig_data_path=None,
                                label_n=True):
    summary = make_summary(PATTERN, p, label, CONFIDENCE)  # summary contains: x, mean_y, std_y, label, n
    summaries.append(summary)
    print(f'--------------------- End section {p.name}')
    print()

# sort data
summaries = sorted(summaries, key=lambda s: s[1][-1], reverse=True)
if not summaries:
    raise SystemExit('No data found')

# print to console
for s in summaries:
    _, y_mean, y_std, label, n = s
    print(label)
    print(y_mean)
    print(y_std)
    print()

# plot
fig = make_summary_fig(summaries,
                       ylabel=' '.join([i.capitalize() for i in PATTERN.split('_')]),
                       title=TITLE,
                       figsize=FIG_SIZE,
                       ylims=Y_LIMS,
                       legend_labels=LABELS,
                       legend_loc='best',
                       )
fig.show()