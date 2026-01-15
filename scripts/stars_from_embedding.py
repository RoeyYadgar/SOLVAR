import os
import pickle

import click
import numpy as np
from matplotlib import pyplot as plt

from solvar.utils import sub_starfile


@click.command()
@click.option("-i", "--input-star", type=str, help="Path to input star file")
@click.option("-o", "output_dir", type=str, help="Directory to save sub star files")
@click.option("--embedding-path", type=str, help="Path to pkl file with algorithm embedding output")
@click.option("--centers-path", type=str, help="Path to pkl file with analysis script centers output")
@click.option("--num-neighbors", "-n", type=int, default=5000, help="Number of neighbors of each center")
@click.option(
    "--skip-index",
    type=str,
    help=(
        "Path to txt file with indeces which were removed from the processed star "
        "- to preserve index of original star"
    ),
)
def create_sub_starfiles_from_embedding(
    input_star, embedding_path, centers_path, output_dir, num_neighbors=5000, skip_index=None
):
    with open(embedding_path, "rb") as f:
        embedding = pickle.load(f)

    with open(centers_path, "rb") as f:
        centers = pickle.load(f)["cluster_coords"]

    os.makedirs(output_dir, exist_ok=True)

    coords = embedding["coords_est"]
    coords_covar_inv = embedding["coords_covar_inv_est"]

    if skip_index is not None:
        pad = np.loadtxt(skip_index)
        coords_new = np.zeros((coords.shape[0] + len(pad), coords.shape[1]))
        coords_covar_inv_new = np.zeros((coords.shape[0] + len(pad), coords.shape[1], coords.shape[1]))
        mask = np.isin(np.arange(coords_new.shape[0]), pad, invert=True)
        coords_new[mask] = coords
        coords_new[pad.astype(np.int64)] = np.inf
        coords_covar_inv_new[mask] = coords_covar_inv

        coords = coords_new
        coords_covar_inv = coords_covar_inv_new

    for i, center in enumerate(centers):
        mean_centered_coords = coords - center
        dist_to_center = np.sum(
            np.matmul(coords_covar_inv, mean_centered_coords[..., np.newaxis]).squeeze(-1) * mean_centered_coords,
            axis=1,
        )
        dist_to_center[np.isnan(dist_to_center)] = np.inf
        neighbors = np.argsort(dist_to_center)
        neighbors_dist = dist_to_center[neighbors]
        neighbors = neighbors[:num_neighbors]
        sub_starfile(input_star, os.path.join(output_dir, f"sub_star{i}.star"), neighbors)
        fig = plt.figure()
        plt.plot(neighbors_dist)
        plt.ylim(0, np.percentile(neighbors_dist, [95]))
        fig.savefig(os.path.join(output_dir, f"distance_from_center{i}.jpg"))


if __name__ == "__main__":
    create_sub_starfiles_from_embedding()
