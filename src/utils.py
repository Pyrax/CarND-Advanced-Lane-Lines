import matplotlib.pyplot as plt
import math


def plot_images_as_grid(images, labels=None, images_per_row=5, figsize=(10, 6), cmap=None):
    if labels is None:
        labels = []

    rows = math.ceil(len(images) / images_per_row)
    columns = images_per_row

    fig = plt.figure(figsize=figsize)
    for i in range(0, len(images)):
        ax = fig.add_subplot(rows, columns, i + 1)
        if i < len(labels):
            ax.set_title(labels[i])
        ax.imshow(images[i], cmap=cmap)
    plt.tight_layout()
    plt.show()
    return fig


def plot_color_channels(image, labels=None, figsize=(18, 4)):
    if labels is None:
        labels = []

    fig, axes = plt.subplots(1, 3, sharey='all', figsize=figsize)
    for idx, ax in enumerate(axes):
        if idx < len(labels):
            ax.set_title(labels[idx])
        ax.imshow(image[:,:,idx], cmap='gray')
    plt.tight_layout()
    plt.show()
    return fig
