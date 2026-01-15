import numpy as np
import torch
from aspire.utils import grid_3d
from aspire.volume import Volume

from solvar.projection_funcs import centered_fft3, centered_ifft3


def process_volume(vol: Volume, L: int, sigma: float = 0.2):
    if vol.shape[-1] > L:
        vol = vol.downsample(L)

    filter = gen_gaussian_filter(L, sigma)

    filter *= torch.tensor(grid_3d(L, shifted=False, normalized=True)["r"] <= 1)

    vol_tensor = centered_fft3(torch.tensor(vol.asnumpy()))
    vol_tensor = centered_ifft3(filter * vol_tensor).real

    vol = Volume(vol_tensor.numpy())

    return vol


def gen_gaussian_filter(L: int, sigma: float = 0.2, dtype=torch.float32):
    """Returns a 3D Gaussian filter matching the dataset volume size."""
    grid = grid_3d(L, shifted=False, normalized=False)
    xx, yy, zz = (grid["x"], grid["y"], grid["z"])

    gaussian = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * (L * sigma) ** 2))
    return torch.tensor(gaussian, dtype=dtype)
