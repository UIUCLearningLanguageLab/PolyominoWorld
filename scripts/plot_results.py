"""
In order to plot results,
we need to get results form the shared drive.
To look for results on the shared drive, we use ludwig.
We use params.param2requests to tell ludwig which jobs we would like results for.
"""

from typing import Optional, List, Tuple
from pathlib import Path

from ludwig.results import gen_param_paths

from polyominoworld import __name__
from polyominoworld.figs import make_summary_fig
from polyominoworld.summary import make_summary
from polyominoworld.params import param2default, param2requests

# where to look for results
LUDWIG_DATA_PATH: Optional[Path] = Path('/media/ludwig_data')
RUNS_PATH = None  # config.Dirs.runs if using local plot or None if using plot form Ludwig
PATTERN: str = 'acc_avg_train'  # name of performance curve to plot

# figure settings
LABELS: Optional[List[str]] = None  # custom labels for figure legend
FIG_SIZE: Tuple[int, int] = (6, 4)  # in inches
Y_LIMS: Optional[List[float]] = None
Y_LABEL: str = ''
CONFIDENCE: float = 0.95
TITLE = ''


param2requests = {'colors': [('red',), ('blue',)]}  # TODO this is where we request which jobs to plot results for


# collect summaries
summaries = []
project_name = __name__
for p, label in gen_param_paths(project_name,
                                param2requests,
                                param2default,
                                runs_path=RUNS_PATH,
                                ludwig_data_path=LUDWIG_DATA_PATH,
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
                       Y_LABEL,
                       title=TITLE,
                       figsize=FIG_SIZE,
                       ylims=Y_LIMS,
                       legend_labels=LABELS,
                       legend_loc='best',
                       )
fig.show()