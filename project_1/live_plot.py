# -*- coding: utf-8 -*-
"""OMSCS Reinforcement Learning - CS-7642-O03 - Project #1 Solution

This code solves Project #1 from OMSCS Reinforcement Learning - CS-7642-O03,
generating the video referenced in report, which can be seen in the following
Youtube link

    https://youtu.be/mBqyQpL8_Vc

To re-create it, you will have to run 2 commands in parallel: i) one to capture
the live data being produced by the simulation (and either display it in the
screen or save it in `images/animation.mp4`), and ii) other to generate the live
data. To do (i), first run:

    $ python live_plot.py

Then, to generate the live data (ii), then run in parallel:

    $ python final_solution.py --live

The default behaviour of `live_plot.py` program is to save the video in the file
`images/animation.mp4`. If instead you want to see it in the screen, run

    $ python live_plot --show

instead.

Created by Carlos Souza (souza@gatech.edu)
May-2019

"""

import pandas as pd
from sys import platform
if platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pandas.errors import EmptyDataError
import matplotlib.gridspec as gridspec
import os
import argparse


fig = plt.figure(figsize=(16, 9))
spec = gridspec.GridSpec(nrows=4, ncols=3, hspace=.5, wspace=.25)
ax1 = fig.add_subplot(spec[0:2, 0])
ax2 = fig.add_subplot(spec[0:2, 1], sharey=ax1)
ax3 = fig.add_subplot(spec[2:4, 0])
ax4 = fig.add_subplot(spec[2:4, 1], sharey=ax3)
ax5 = fig.add_subplot(spec[2:4, 2])


def animate(i, file_v, file_e, file_error, file_annotation, file_evol):
    if os.path.isfile(file_v) and os.path.isfile(file_e) \
            and os.path.isfile(file_annotation):

        try:
            df_v = pd.read_csv(file_v, index_col=0)
            df_e = pd.read_csv(file_e, index_col=0)
            #df_evol = pd.read_csv(file_evol, index_col=0)
            f = open(file_annotation, "r")
            annotation = f.read()
            f.seek(0, 0)
            line_list = f.readlines()
            curr_state_index = -1
            next_state_index = -1
            if len(line_list) > 4:
                curr_state_index = ord((line_list[3].split(':')[1]).strip()) - 65
                if len((line_list[4].split(':')[1]).strip()) == 1:
                    next_state_index = ord((line_list[4].split(':')[1]).strip()) - 65
                else:
                    next_state_index = -1

            f.close()

            ax1.clear()
            ax2.clear()
            ax3.clear()
            ax4.clear()
            ax5.clear()
            for t in fig.texts:
                t.remove()

            base_color = 'lime'
            max_color = 'darkgreen'
            bg_color = 'ivory'
            factor = 1.5

            # Ax1
            bl1 = ax1.bar(df_v.columns, df_v.iloc[1], color=base_color)
            for i, bar1 in enumerate(bl1):
                amount = df_e.iloc[0, i] * factor if df_e.iloc[0, i] * factor <= 1 else 1
                c = darken_color(base_color, amount)
                if amount == 1:
                    c = max_color
                bar1.set_color(c)

            ax1.plot(df_v.columns, df_v.iloc[0], 'ys')
            ax1.set_title(f'$\lambda = {df_v.index.values[1]}$')
            ax1.set_ylabel('State-value function estimate')
            ax1.set_facecolor(bg_color)

            # Ax2
            bl2 = ax2.bar(df_v.columns, df_v.iloc[2], color=base_color)
            for i, bar2 in enumerate(bl2):
                amount = df_e.iloc[1, i] * factor if df_e.iloc[1, i] * factor <= 1 else 1
                c = darken_color(base_color, amount)
                if amount == 1:
                    c = max_color
                bar2.set_color(c)

            ax2.plot(df_v.columns, df_v.iloc[0], 'ys')
            ax2.set_title(f'$\lambda = {df_v.index.values[2]}$')
            ax2.set_ylabel('State-value function estimate')
            ax2.set_facecolor(bg_color)

            # Ax3
            ax3.bar(df_e.columns, df_e.iloc[0], color='royalblue')
            ax3.set_title(f'$\lambda = {df_e.index.values[0]}$')
            ax3.set_ylabel('Eligibility')
            ax3.set_facecolor(bg_color)

            # Ax4
            ax4.bar(df_e.columns, df_e.iloc[1], color='r')
            ax4.set_title(f'$\lambda = {df_e.index.values[1]}$')
            ax4.set_ylabel('Eligibility')
            ax4.set_facecolor(bg_color)

            # Ax5
            if os.path.isfile(file_error):
                df_error = pd.read_csv(file_error, index_col=0)
                ax5.plot(df_error.iloc[:, 0], 'o-',
                         label=f'$\lambda = {df_error.columns[0]}$',
                         color='royalblue', markersize=2)
                ax5.plot(df_error.iloc[:, 1], 'o-',
                         label=f'$\lambda = {df_error.columns[1]}$',
                         color='red', markersize=2)
                ax5.set_title('RMS Error vs. true state-values')
                ax5.set_ylabel('RMS Error')
                ax5.set_xlabel('Episodes')
                ax5.legend(frameon=False)
                ax5.set_facecolor(bg_color)

            # Annotation info
            plt.figtext(.67, .62, annotation, fontsize=22, linespacing=1.5)

        except EmptyDataError:
            pass


def darken_color(color, amount=0.5):
    """
    Darkens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> darken_color('g', 0.3)
    >> darken_color('#F034A3', 0.6)
    >> darken_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - (amount + 1) * (1 - c[1]), c[2])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Project #1 animation.')
    parser.add_argument('-s', '--show', action="store_true", default=False,
                        help='Use this flag to show video animation on screen'
                             'instead of saving it to file. Default is to save'
                             'to "animation.mp4".')
    args = parser.parse_args()

    file_v = 'images/df_v.csv'
    file_e = 'images/df_e.csv'
    file_error = 'images/df_error.csv'
    file_evol = 'images/df_evol.csv'
    file_annotation = 'images/annotation.txt'
    try:
        os.remove(file_v)
        os.remove(file_e)
        os.remove(file_error)
        os.remove(file_evol)
        os.remove(file_annotation)
    except OSError:
        pass

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Carlos Souza'), bitrate=1800)

    ani = animation.FuncAnimation(fig, animate, interval=10, frames=10000,
                                  fargs=(file_v, file_e, file_error,
                                         file_annotation, file_evol, ))
    if args.show:
        plt.show()
    else:
        ani.save('images/animation.mp4', writer=writer)
