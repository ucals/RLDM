import pandas as pd
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

            #Ax1
            bl1 = ax1.bar(df_v.columns, df_v.iloc[1], color='cornflowerblue')
            if curr_state_index != -1:
                bl1[curr_state_index].set_color('blue')
            if next_state_index != -1:
                bl1[next_state_index].set_color('mediumblue')

            ax1.plot(df_v.columns, df_v.iloc[0], 'ys')
            ax1.set_title(f'$\lambda = {df_v.index.values[1]}$')
            ax1.set_ylabel('State-value function estimate')

            #Ax2
            bl2 = ax2.bar(df_v.columns, df_v.iloc[2], color='cornflowerblue')
            if curr_state_index != -1:
                bl2[curr_state_index].set_color('blue')
            if next_state_index != -1:
                bl2[next_state_index].set_color('mediumblue')

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
            if os.path.isfile(file_error):
                df_error = pd.read_csv(file_error, index_col=0)
                ax5.plot(df_error.iloc[:, 0], label=f'$\lambda = {df_error.columns[0]}$')
                ax5.plot(df_error.iloc[:, 1], label=f'$\lambda = {df_error.columns[1]}$')
                ax5.set_title('RMS Error vs. true state-values')
                ax5.set_ylabel('RMS Error')
                ax5.set_xlabel('Episodes')
                ax5.legend(frameon=False)

            #Annotation info
            plt.figtext(.67, .62, annotation, fontsize=22, linespacing=1.5)

        except EmptyDataError:
            pass


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
