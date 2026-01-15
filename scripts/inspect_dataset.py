import os
import pickle

import click
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from solvar.fsc_utils import rpsd


@click.command()
@click.option("-i", "--dataset-path", type=str)
@click.option("-o", "--output-path", type=str)
def inspect_dataset(dataset_path, output_path):
    dataset = pickle.load(open(dataset_path, "rb"))
    L = dataset.resolution
    sample_images = dataset.images[:5].transpose(0, 1).reshape(L, -1)

    gt_datapath = os.path.join(os.path.split(dataset_path)[0], "gt_data.pkl")
    gt_data = pickle.load(open(gt_datapath, "rb"))
    is_vectors_gt = gt_data.eigenvecs is not None

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(2, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.imshow(sample_images)
    cbar = fig.colorbar(ax1.imshow(sample_images), ax=ax1, orientation="vertical")
    cbar.set_label("Intensity")
    ax1.set_title("Image samples")

    ax2.plot(dataset.signal_rpsd)
    ax2.set_yscale("log")
    ax2.set_title("Signal RPSD estimate")

    ax3.plot(dataset.radial_filters_gain)
    ax3.set_yscale("log")
    ax3.set_title("Filter radial gain")

    if is_vectors_gt:
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.plot(rpsd(*gt_data.eigenvecs.reshape(-1, L, L, L)).T)
        ax4.set_yscale("log")
        ax4.set_title("Groundtruth eigen volumes RPSD")

    fig.savefig(output_path)


if __name__ == "__main__":
    inspect_dataset()
