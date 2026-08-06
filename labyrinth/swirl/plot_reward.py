#!/usr/bin/env python3
"""
Plot learned labyrinth reward maps using maze structure.

Expects the result npz from run_labyrinth_history.py, which already stores:
- R_avg_ks : (K, S) action-averaged reward map per mode
This is enough for visualization without reconstructing the reward net.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors

from plot_labyrinth import PlotMazeFunction, normalize


DEFAULT_COLORS = [
    (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    (0.814, 0.661, 0.885, 0.9),
    (0.85, 0.5, 0.15, 0.95),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_npz', type=str, required=True,
                    help='Output npz from run_labyrinth_history.py')
    ap.add_argument('--maze_info', type=str, default='../data/maze_info.npz')
    ap.add_argument('--out_pdf', type=str, default=None)
    ap.add_argument('--mode_names', type=str, nargs='*', default=None,
                    help='Optional names for modes. Default: Mode 0, Mode 1, ...')
    return ap.parse_args()


def main():
    args = parse_args()
    z = np.load(args.result_npz, allow_pickle=True)
    if 'R_avg_ks' not in z.files:
        raise ValueError('Expected R_avg_ks in result npz.')
    R_avg_ks = np.asarray(z['R_avg_ks'], dtype=float)  # (K,S)
    K, S = R_avg_ks.shape

    maze = np.load(args.maze_info, allow_pickle=True)
    m_wa, m_ru, m_xc, m_yc = maze['m_wa'], maze['m_ru'], maze['m_xc'], maze['m_yc']

    mode_names = args.mode_names
    if mode_names is None or len(mode_names) != K:
        mode_names = [f'Mode {k}' for k in range(K)]

    fig, axes = plt.subplots(1, K, figsize=(6 * K, 6), dpi=300)
    if K == 1:
        axes = [axes]

    norm = plt.Normalize(vmin=0, vmax=1)

    for k in range(K):
        r = normalize(np.asarray(R_avg_ks[k], dtype=float))
        color = DEFAULT_COLORS[k % len(DEFAULT_COLORS)]
        PlotMazeFunction(
            r,
            mode_names[k],
            m_wa, m_ru, m_xc, m_yc,
            numcol='blue',
            figsize=6,
            selected_color=color,
            axes=axes[k],
        )

        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            'custom_cmap', [(1, 1, 1, 1), color]
        )
        sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
        divider = make_axes_locatable(axes[k])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(sm, cax=cax, ticks=[0, 1])
        cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()

    out_pdf = args.out_pdf
    if out_pdf is None:
        out_pdf = str(Path(args.result_npz).with_suffix('')) + '_reward_maps.pdf'
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f'[ok] saved reward maps to {out_pdf}')


if __name__ == '__main__':
    main()
