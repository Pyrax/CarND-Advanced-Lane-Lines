import matplotlib.pyplot as plt
import math


def plot_images_as_grid(images, images_per_row=5, figsize=(10, 6)):
    rows = math.ceil(len(images) / images_per_row)
    columns = images_per_row

    fig = plt.figure(figsize=figsize)
    for i in range(0, len(images)):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(images[i])
    plt.tight_layout()
    plt.show()
