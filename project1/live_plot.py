import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from pandas.errors import EmptyDataError
import matplotlib.gridspec as gridspec


fig = plt.figure(figsize=(16, 9))
spec = gridspec.GridSpec(nrows=4, ncols=3, hspace=.5)
ax1 = fig.add_subplot(spec[0:2, 0])
ax2 = fig.add_subplot(spec[0:2, 1], sharey=ax1)
ax3 = fig.add_subplot(spec[2:4, 0])
ax4 = fig.add_subplot(spec[2:4, 1], sharey=ax3)
ax5 = fig.add_subplot(spec[1:3, 2])


def animate(i, file_v, file_e, file_error, file_annotation):
    try:
        df_v = pd.read_csv(file_v, index_col=0)
        df_e = pd.read_csv(file_e, index_col=0)
        df_error = pd.read_csv(file_error, index_col=0)
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

        #Ax1
        bl1 = ax1.bar(df_v.columns, df_v.iloc[1])
        if curr_state_index != -1:
            bl1[curr_state_index].set_color('r')
        if next_state_index != -1:
            bl1[next_state_index].set_color('g')

        ax1.plot(df_v.columns, df_v.iloc[0], 'ys')
        ax1.set_title(f'$\lambda = {df_v.index.values[1]}$')
        ax1.set_ylabel('State-value function estimate')

        #Ax2
        bl2 = ax2.bar(df_v.columns, df_v.iloc[2])
        if curr_state_index != -1:
            bl2[curr_state_index].set_color('r')
        if next_state_index != -1:
            bl2[next_state_index].set_color('g')

        ax2.plot(df_v.columns, df_v.iloc[0], 'ys')
        ax2.set_title(f'$\lambda = {df_v.index.values[2]}$')
        ax2.set_ylabel('State-value function estimate')

        #Ax3
        ax3.bar(df_e.columns, df_e.iloc[0])
        ax3.set_title(f'$\lambda = {df_e.index.values[0]}$')
        ax3.set_ylabel('Eligibility')

        #Ax4
        ax4.bar(df_e.columns, df_e.iloc[1])
        ax4.set_title(f'$\lambda = {df_e.index.values[1]}$')
        ax4.set_ylabel('Eligibility')

        #Ax5
        ax5.plot(df_error)
        ax5.set_title('RMS Error vs. true state-values')
        ax5.set_ylabel('RMS Error')
        ax5.set_xlabel('Episodes')
        #ax5.legend()

        #Annotation info
        plt.figtext(.67, .75, annotation, fontsize=14, linespacing=1.5)

    except EmptyDataError:
        pass


if __name__ == '__main__':
    file_v = 'images/df_v.csv'
    file_e = 'images/df_e.csv'
    file_error = 'images/df_error.csv'
    file_annotation = 'images/annotation.txt'
    ani = animation.FuncAnimation(fig, animate, interval=10,
                                  fargs=(file_v, file_e, file_error,
                                         file_annotation, ))
    plt.show()
