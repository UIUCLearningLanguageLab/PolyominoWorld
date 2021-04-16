import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple, Union, Optional

from polyominoworld import configs


def rank_label_for_legend_order(label:  str,
                                ) -> int:
    """assign rank to a label, for ordering labels in figure legend"""

    rank = 100

    for label_part in label.split('\n'):
        if label_part == 'load_from_checkpoint=none':
            print('found')
            rank = 0

        # shapes
        for n, shape in enumerate(configs.World.master_shapes):
            if shape in label_part:
                rank = n

        # colors
        for n, color in enumerate(configs.World.master_colors):
            if color in label_part:
                rank = n

        # positions
        for n, half in enumerate(['lower', 'upper']):
            if half in label_part:
                rank = n

        # variants
        for n, variants in enumerate(['half1', 'half2']):
            if variants in label_part:
                rank = n

    print(f'Assigned label="{label}" rank={rank} in legend order')
    return rank


def make_summary_fig(summaries: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]],
                     y_label: str,
                     x_label: str,
                     title: str = '',
                     palette_ids: List[int] = None,
                     figsize: Tuple[int, int] = None,
                     y_lims: List[float] = None,
                     x_lims: List[int] = None,
                     log_y: bool = False,
                     start_x_at_zero: bool = False,
                     y_grid: bool = False,
                     legend_labels: Union[None, list] = None,
                     legend_loc: str = 'lower right',
                     verbose: bool = False,
                     ):
    # setting up the matplotlib figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=configs.Figs.dpi)
    plt.title(title)
    ax.set_xlabel(x_label, fontsize=configs.Figs.ax_label_fs)
    ax.set_ylabel(y_label, fontsize=configs.Figs.ax_label_fs)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if y_grid:
        ax.yaxis.grid(True)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    if log_y:
        ax.set_yscale('log')
    if start_x_at_zero:
        ax.set_xlim(xmin=0, xmax=summaries[0][0][-1])
    if x_lims is not None:
        ax.set_xlim(x_lims)

    # palette - assign each configuration a different color
    num_summaries = len(summaries)
    palette = np.asarray(sns.color_palette('hls', num_summaries))
    if palette_ids is not None:
        colors = iter(palette[palette_ids])
    else:
        colors = iter(palette)

    if legend_labels is not None:
        legend_labels = iter(legend_labels)

    # plot summary
    max_ys = []
    for x, y_mean, h, label, n in summaries:  # there is one summary for each configuration

        # x is a list of x-axis values
        # y is a list of y-axi values
        # h is the margin of error around y (resulting in the the 95% confidence interval)
        max_ys.append(max(y_mean))

        if legend_labels is not None:
            try:
                label = next(legend_labels)
            except StopIteration:
                raise ValueError('Not enough values in ALTERNATIVE_LABELS')

        try:
            color = next(colors)
        except StopIteration:
            raise ValueError('Not enough values in PALETTE_IDS')

        if verbose:
            for mean_i, std_i in zip(y_mean, h):
                print(f'mean={mean_i:>6.2f} h={std_i:>6.2f}')

        # plot the lines
        ax.plot(x, y_mean, '-',
                linewidth=configs.Figs.lw,
                color=color,
                label=label,  # legend label
                zorder=3 if n == 8 else 2)

        # plots the margin of error (shaded region)
        ax.fill_between(x, y_mean + h, y_mean - h, alpha=0.2, color=color)

    # legend
    if title:
        plt.legend(fontsize=configs.Figs.leg_fs, frameon=False, loc=legend_loc, ncol=1)
    else:
        plt.legend(bbox_to_anchor=(0.5, 1.0),
                   borderaxespad=1.0,
                   fontsize=configs.Figs.leg_fs,
                   frameon=False,
                   loc='lower center',
                   ncol=3,
                   )

    plt.tight_layout()
    return fig


def make_y_label(pattern: str,
                 ) -> str:
    """make more readable y-axis label for figure"""
    res = ''
    for pattern_part in pattern.split('_')[::-1]:
        try:
            line = {'acc': 'Accuracy',
                    'cost': 'Error',
                    'train': 'TrainData',
                    'valid': 'TestData',
                    }[pattern_part]
        except KeyError:
            line = pattern_part
        res += line.capitalize() + ' '
    return res


def plot_hidden_weights_analysis(mat: np.array,
                                 y_tick_labels: Optional[List[str]] = None,
                                 x_tick_labels: Optional[List[str]] = None,
                                 title: Optional[str] = None,
                                 max_x: int = 4,
                                 dpi: int = 192 // 2
                                 ):
    fig, (top, bottom) = plt.subplots(2, 2, figsize=(6, 6), dpi=dpi)
    if title is not None:
        plt.title(title)

    for (ax1, ax2), mat in zip([top, bottom],
                               [mat.round(1), np.rint(mat)]):
        # heatmap
        ax1.imshow(mat,
                   aspect='equal',
                   cmap=plt.get_cmap('jet'),
                   interpolation='nearest',
                   vmin=-max_x,
                   vmax=+max_x,
                   )
        # tick labels
        if x_tick_labels is not None and y_tick_labels is not None:
            ax1.set_xticks([n for n, _ in enumerate(x_tick_labels)])
            ax1.set_yticks([n for n, _ in enumerate(y_tick_labels)])
            ax1.set_xticklabels(x_tick_labels)
            ax1.set_yticklabels(y_tick_labels)
        # remove tick lines
        lines = (ax1.xaxis.get_ticklines() +
                 ax1.yaxis.get_ticklines())
        plt.setp(lines, visible=False)

        ax2.hist(mat.flatten(), bins=16)
        ax2.set_xlim([-max_x, max_x])

    plt.show()


def plot_state_analysis(mat: np.array,
                        y_tick_labels: Optional[List[str]] = None,
                        x_tick_labels: Optional[List[str]] = None,
                        title: str = '',
                        dpi: int = 192 // 2
                        ):
    fig, axes = plt.subplots(1, 2, figsize=(6, 6), dpi=dpi)

    # heatmaps
    for ax, states_name, mat in zip(axes,
                                    ['total', 'unique'],
                                    [mat, np.unique(mat, axis=0)]):
        ax.set_title(title + '\n' + f'num {states_name} states={len(mat)}')
        ax.imshow(mat,
                  aspect='equal',
                  cmap=plt.get_cmap('jet'),
                  interpolation='nearest')
        # tick labels
        if x_tick_labels is not None and y_tick_labels is not None:
            ax.set_xticks([n for n, _ in enumerate(x_tick_labels)])
            ax.set_yticks([n for n, _ in enumerate(y_tick_labels)])
            ax.set_xticklabels(x_tick_labels)
            ax.set_yticklabels(y_tick_labels)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        # remove tick lines
        lines = (ax.xaxis.get_ticklines() +
                 ax.yaxis.get_ticklines())
        plt.setp(lines, visible=False)

    plt.show()