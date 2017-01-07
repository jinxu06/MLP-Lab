import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc
rc('text', usetex=True)

def plot_stats(axis, stats, keys, labels, stats_interval=1, legend=None):
    if legend is None:
        for label in labels:
            axis.plot(np.arange(1, stats.shape[0]) * stats_interval,stats[1:, keys[label]], label=label)
    else:
        for label in labels:
            axis.plot(np.arange(1, stats.shape[0]) * stats_interval,stats[1:, keys[label]], label=legend)
    return axis

def plot_training_evolvement(stats, keys):
    fig = plt.figure(figsize=(12,4))
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    plot_stats(ax1, stats, keys, ['acc(train)', 'acc(valid)'])
    ax1.set_xlabel("epoch number")
    ax1.set_ylabel("accuracy")
    ax1.set_title("accuracy evolvement", fontsize=10)
    ax1.legend(loc=0)

    plot_stats(ax2, stats, keys, ['error(train)', 'error(valid)'])
    ax2.set_xlabel("epoch number")
    ax2.set_ylabel("error function value")
    ax2.set_title("error evolvement", fontsize=10)
    ax2.legend(loc=0)

    return fig

def plot_setting_compare(stats_arr, keys, legends):

    fig1 = plt.figure(figsize=(12,4))
    ax1 = fig1.add_subplot(1,2,1)
    ax2 = fig1.add_subplot(1,2,2)

    for i, stats in enumerate(stats_arr):

        plot_stats(ax1, stats, keys, ['acc(train)'], legend=legends[i])
        ax1.set_xlabel('epoch number')
        ax1.set_ylabel('accuracy')
        ax1.set_title('training accuracy evolvement', fontsize=10)
        ax1.legend(loc=0)

        plot_stats(ax2, stats, keys, ['acc(valid)'], legend=legends[i])
        ax2.set_xlabel("epoch number")
        ax2.set_ylabel("accuracy")
        ax2.set_title("validation accuracy evolvement", fontsize=10)
        ax2.legend(loc=0)


    fig2 = plt.figure(figsize=(12,4))
    ax1 = fig2.add_subplot(1,2,1)
    ax2 = fig2.add_subplot(1,2,2)

    for i, stats in enumerate(stats_arr):

        plot_stats(ax1, stats, keys, ['error(train)'], legend=legends[i])
        ax1.set_xlabel('epoch number')
        ax1.set_ylabel('error')
        ax1.set_title('training error evolvement', fontsize=10)
        ax1.legend(loc=0)

        plot_stats(ax2, stats, keys, ['error(valid)'], legend=legends[i])
        ax2.set_xlabel("epoch number")
        ax2.set_ylabel("error")
        ax2.set_title("validation error evolvement", fontsize=10)
        ax2.legend(loc=0)

    return (fig1, fig2)


def show_batch_of_images(img_batch, fig_size=(3, 3), num_rows=None):
    fig = plt.figure(figsize=fig_size)
    batch_size, im_height, im_width = img_batch.shape
    if num_rows is None:
        # calculate grid dimensions to give square(ish) grid
        num_rows = int(batch_size**0.5)
    num_cols = int(batch_size * 1. / num_rows)
    if num_rows * num_cols < batch_size:
        num_cols += 1
    # intialise empty array to tile image grid into
    tiled = np.zeros((im_height * num_rows, im_width * num_cols))
    # iterate over images in batch + indexes within batch
    for i, img in enumerate(img_batch):
        # calculate grid row and column indices
        r, c = i % num_rows, i // num_rows
        tiled[r * im_height:(r + 1) * im_height, 
              c * im_height:(c + 1) * im_height] = img
    ax = fig.add_subplot(111)
    ax.imshow(tiled, cmap='Greys', vmin=0., vmax=1.)
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig, ax

