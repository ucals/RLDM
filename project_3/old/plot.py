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


def plot_error(ax, df):
    ax.plot(df['timestep'], df['error'])
    ax.set_ylim(0, 0.5)
    ax.set_title('Evolution of error in state S')
    ax.set_xlabel('Timestep', size='small')
    ax.set_ylabel('Error', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)


def plot_alpha(ax, df):
    ax.plot(df['timestep'], df['alpha'])
    ax.set_title('Evolution of learning rate')
    ax.set_xlabel('Timestep', size='small')
    ax.set_ylabel('α', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)


def plot_epsilon(ax, df):
    ax.plot(df['timestep'], df['epsilon'])
    ax.set_title('Evolution of random actions')
    ax.set_xlabel('Timestep', size='small')
    ax.set_ylabel('ε', size='small')
    ax.tick_params(axis='both', which='major', labelsize=10)
    format_axis(ax)


def annotate_info(ax, df, position=(0.02, 0.9)):
    info = f"Timestep {df['timestep'].iloc[-1] + 1:.0f}: " \
           f"ε {df['epsilon'].iloc[-1]:0.3f}, " \
           f"α {df['alpha'].iloc[-1]:0.3f} \n" \
           f"Error {df['error'].tail(100).min():0.4f} "
    ax.annotate(info, xy=position, xycoords='axes fraction', size='large')


def format_axis(axis):
    axis.grid(b=True, which='major', color='#cccccc', linewidth=0.4, linestyle='--')
    axis.set_facecolor('#f9f9f9')
    plt.setp(axis.spines.values(), color='#cccccc')
    plt.setp([axis.get_xticklines(), axis.get_yticklines()], color='#cccccc')


def animate(i, filename):
    try:
        df = pd.read_csv(filename)
        ax1.clear()
        ax2.clear()
        ax3.clear()
        plot_error(ax1, df)
        plot_alpha(ax2, df)
        plot_epsilon(ax3, df)
        #annotate_info(ax1, df, position=(0.5, 0.9))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

    except EmptyDataError:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot real-time scores.')
    parser.add_argument('filename', metavar='f', type=str,
                        help='csv filename with scores')
    parser.add_argument('-w', '--watermark', action='store_true',
                        help='plot with watermark')
    args = parser.parse_args()

    if args.watermark:
        font = {'color':  'gray', 'weight': 'normal', 'size': 80, 'alpha': 0.2}
        fig.text(0.15, 0.6, 'Carlos Souza', fontdict=font)

    ani = animation.FuncAnimation(fig, animate, interval=2000,
                                  fargs=(args.filename, ))
    plt.show()
