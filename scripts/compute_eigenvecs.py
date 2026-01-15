import os

import click
import numpy as np
from aspire.volume import Volume

from solvar.utils import volsCovarEigenvec


@click.command()
@click.option("--input-dir", "-i", help="Input to volume directory", required=True)
@click.option("--output", "-o", help="Output mrc path", required=True)
@click.option("--rank", "-r", type=int, help="Number of leading eigenvecs to compute", default=None)
def compute_covar_eigenvecs(input_dir, output, rank):
    """Compute the eigenvectors of the covariance matrix of the input volumes.

    Assumes volumes are uniformly distriubted from states in inputer directory.
    """
    volumes_path = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".mrc")]
    volume_size = Volume.load(volumes_path[0]).shape[-1]
    volumes = Volume(np.zeros((len(volumes_path), volume_size, volume_size, volume_size)))
    for i in range(len(volumes_path)):
        volumes[i] = Volume.load(volumes_path[i])
    cov_eigenvecs = volsCovarEigenvec(volumes)
    if rank is not None:
        cov_eigenvecs = cov_eigenvecs[:rank]

    Volume(cov_eigenvecs.reshape(-1, volume_size, volume_size, volume_size)).save(output)


if __name__ == "__main__":
    compute_covar_eigenvecs()
