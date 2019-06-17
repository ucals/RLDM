#!/anaconda3/bin/python

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from pandas.errors import EmptyDataError


#style.use('fivethirtyeight')
fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((3, 2), (2, 0))
ax3 = plt.subplot2grid((3, 2), (2, 1))


def plot_avg_scores(ax, df):
    ax.plot(df['episode'], df['average'])
    ax.set_title('Evolution of average scores from past 100 episodes')
    ax.set_xlabel('Episode', size='small')
    ax.set_ylabel('Score', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)
    if ax.get_ylim()[1] > 195:
        ax.axhline(195, color='red', linewidth=0.8, linestyle='--')


def annotate_info(ax, df, position=(0.02, 0.9)):
    info = f"Episode {df['episode'].iloc[-1]:.0f}: " \
        f"score {df['score'].iloc[-1]:.0f}, " \
        f"epsilon {df['epsilon'].iloc[-1]:0.3f} \n" \
        f"Past {df.shape[0]} runs: " \
        f"min {df['score'].tail(100).min():.0f}, " \
        f"max {df['score'].tail(100).max():.0f}, " \
        f"avg {df['score'].tail(100).mean():.0f}"
    ax.annotate(info, xy=position, xycoords='axes fraction', size='large')


def plot_scores(ax, df):
    ax.plot(df['episode'], df['score'])
    ax.set_title('Evolution of scores')
    ax.set_xlabel('Episode', size='small')
    ax.set_ylabel('Score', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)
    if ax.get_ylim()[1] > 200:
        ax.axhline(200, color='red', linewidth=0.8, linestyle='--')


def plot_scores_and_avg(ax, df):
    ax.plot(df['episode'], df['score'])
    ax.plot(df['episode'], df['average'])
    ax.set_title('Evolution of scores')
    ax.set_xlabel('Episode', size='small')
    ax.set_ylabel('Score', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)
    if ax.get_ylim()[1] > 200:
        ax.axhline(200, color='red', linewidth=0.8, linestyle='--')


def plot_epsilon(ax, df):
    ax.plot(df['episode'], df['epsilon'])
    ax.set_title('Evolution of random actions')
    ax.set_xlabel('Episode', size='small')
    ax.set_ylabel('Epsilon', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)


def plot_avg_q_values(ax, df):
    ax.plot(df['episode'], df['avg_q_values'])
    ax.set_title('Evolution of average Q values')
    ax.set_xlabel('Episode', size='small')
    ax.set_ylabel('Average of Q values', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)


def format_axis(axis):
    axis.grid(b=True, which='major', color='#cccccc', linewidth=0.4, linestyle='--')
    axis.set_facecolor('#f9f9f9')
    plt.setp(axis.spines.values(), color='#cccccc')
    plt.setp([axis.get_xticklines(), axis.get_yticklines()], color='#cccccc')


def animate(i, filename, window):
    try:
        df = pd.read_csv(filename)
        if window is not None:
            df = df.tail(window)

        ax1.clear()
        ax2.clear()
        ax3.clear()
        #plot_avg_scores(ax1, df)
        plot_avg_q_values(ax2, df)
        #plot_scores(ax2, df)
        plot_scores_and_avg(ax1, df)
        plot_epsilon(ax3, df)
        annotate_info(ax1, df, position=(0.5, 0.02))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
    except EmptyDataError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot real-time scores.')
    parser.add_argument('filename', metavar='f', type=str,
                        help='csv filename with scores')
    parser.add_argument('-w', '--window', metavar='w', type=int,
                        help='window to plot')
    args = parser.parse_args()

    #window = 100
    ani = animation.FuncAnimation(fig, animate, interval=2000,
                                  fargs=(args.filename, args.window, ))
    plt.show()
