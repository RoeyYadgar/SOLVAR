import gzip
import os
import shutil
import unittest

import numpy as np
import requests
import torch
from aspire.volume import LegacyVolume, Volume
from matplotlib import pyplot as plt

from solvar.dataset import CovarDataset
from solvar.mean import reconstruct_mean, reconstruct_mean_from_halfsets
from solvar.nufft_plan import NufftPlanDiscretized, NufftSpec
from solvar.projection_funcs import centered_fft3
from solvar.source import SimulatedSource
from solvar.utils import saveVol

from .utils import process_volume


def download_mrc(emd_id: int, output_path: str):
    # Build URL
    base_ftp = "https://ftp.ebi.ac.uk/pub/databases/emdb/structures"
    url = f"{base_ftp}/EMD-{emd_id}/map/emd_{emd_id}.map.gz"
    output_path = os.path.abspath(output_path)
    output_gz = f"{output_path}.gz"
    output_dir = os.path.dirname(output_path)

    # Make directory if needed
    os.makedirs(output_dir, exist_ok=True)

    print(f"Downloading from {url} ...")

    # Stream download so large files don't overwhelm memory
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    with open(output_gz, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)

    with gzip.open(output_gz, "rb") as f_in:
        with open(output_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    os.remove(output_gz)

    print(f"Saved to {output_path}")
    return output_path


def display_projections(volume):
    volume = volume.squeeze()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    volume = volume.squeeze()
    axes = ["X", "Y", "Z"]
    for i, ax in enumerate(axs):
        if i == 0:
            proj = volume.sum(dim=0).cpu().numpy()
        elif i == 1:
            proj = volume.sum(dim=1).cpu().numpy()
        else:
            proj = volume.sum(dim=2).cpu().numpy()
        im = ax.imshow(proj, cmap="viridis")
        ax.set_title(f"Projection along {axes[i]}")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    return fig


class TestYourClassName(unittest.TestCase):

    def _gen_rand_volume(self):
        voxels = LegacyVolume(L=self.L, C=1, K=64, dtype=self.dtype_np).generate()
        # Create a smooth radially symmetric volume: 1 at center, 0 for radius >= L/2
        L = self.L
        # Make 3D coords (centered)
        grid = np.arange(L) - (L - 1) / 2
        zz, yy, xx = np.meshgrid(grid, grid, grid, indexing="ij")
        rr = np.sqrt(xx**2 + yy**2 + zz**2)
        # Scale radius to [0,1] (r=0 center, r=0.5*L edge), use smooth cutoff (e.g. cosine taper)
        rnorm = (rr / (L / 2)) * 3
        smooth = np.zeros_like(rnorm)
        inside = rnorm <= 1
        # Use raised cosine (smoothly goes from 1 to 0)
        smooth[inside] = 0.5 * (1 + np.cos(np.pi * rnorm[inside]))
        voxels = voxels * smooth[np.newaxis, ...].astype(self.dtype_np)
        return voxels

    def _gen_volume(self):
        vol = Volume.load("/home/ry295/pi_data/igg_1d/vols/128_org/000.mrc", dtype=self.dtype_np)
        vol = process_volume(vol, self.L, sigma=0.2)
        return vol

    def _gen_dataset(self, vol: Volume, nufft_type: str = "nufft"):
        if nufft_type == "discretized":
            nufft_spec = NufftSpec(
                NufftPlanDiscretized, sz=(self.L,) * 3, upsample_factor=self.upsampling_factor, mode="nearest"
            )
        else:
            nufft_spec = None
        src = SimulatedSource(n=self.n, vols=vol, noise_var=0, whiten=False, nufft_spec=nufft_spec)
        return CovarDataset(src, noise_var=0, apply_preprocessing=False)

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)
        self.L = 128
        self.n = 30000
        self.upsampling_factor = 2
        self.dtype = torch.float64
        self.dtype_np = np.float64

    def tearDown(self):
        # Clean up after tests
        pass

    def _test_reconstruct_mean_clean_dataset(self, source_nufft_type: str):
        source_vol = self._gen_volume()
        dataset = self._gen_dataset(source_vol, nufft_type=source_nufft_type)
        source_vol = torch.tensor(source_vol.asnumpy())
        disc_nufft = source_nufft_type == "discretized"

        reconstructed_mean, rhs, lhs = reconstruct_mean(
            dataset, upsampling_factor=self.upsampling_factor, return_lhs_rhs=True, do_grid_correction=(not disc_nufft)
        )
        reconstructed_mean = reconstructed_mean.to("cpu")
        rhs = rhs.to("cpu")
        lhs = lhs.to("cpu")

        source_vol_fourier = centered_fft3(source_vol).squeeze()
        reconstructed_mean_fourier = centered_fft3(reconstructed_mean)

        if self.L % 2 == 0:
            # For even size we do not recover the negative fourier elements that don't have a positive counterpart
            # i.e. -N/2, -N/2 + 1, ..., N/2-1 we disregard -N/2 from the mask
            rhs = rhs[1:, 1:, 1:]
            lhs = lhs[1:, 1:, 1:]

            source_vol_fourier = source_vol_fourier[1:, 1:, 1:]
            reconstructed_mean_fourier = reconstructed_mean_fourier[1:, 1:, 1:]

        if disc_nufft:
            # When the source images come from actual NUFFT we cannot guarntee per-voxel accuracy
            torch.testing.assert_close(
                reconstructed_mean_fourier, source_vol_fourier, rtol=1e-8, atol=1e-4 * source_vol_fourier.abs().max()
            )

        regularized_reconstructed_mean = reconstruct_mean_from_halfsets(
            dataset, upsampling_factor=self.upsampling_factor, do_grid_correction=(not disc_nufft)
        )
        regularized_reconstructed_mean = regularized_reconstructed_mean.to("cpu")
        relative_error = torch.norm(reconstructed_mean - source_vol) / torch.norm(source_vol)
        relative_error_regularized = torch.norm(regularized_reconstructed_mean - source_vol) / torch.norm(source_vol)
        print(f"Relative error {relative_error} (unregularized) {relative_error_regularized} (regularized)")

        # -----For debugging - Inspect volumes-----------
        debug_vols = [
            (reconstructed_mean, "reconstructed_unreg.mrc"),
            (regularized_reconstructed_mean, "reconstructed_reg.mrc"),
            (source_vol, "source_vol.mrc"),
        ]
        for vol, fname in debug_vols:
            saveVol(vol, fname)

        # break point here
        for _, fname in debug_vols:
            os.remove(fname)
        # --------------------------------

        relative_error_threshold = 1e-4 if disc_nufft else 2e-2
        torch.testing.assert_close(
            relative_error, torch.zeros_like(relative_error), atol=relative_error_threshold, rtol=0
        )

    def test_reconstruct_mean_clean_dataset_disc(self):
        self._test_reconstruct_mean_clean_dataset("discretized")

    def test_reconstruct_mean_clean_dataset_nufft(self):
        self._test_reconstruct_mean_clean_dataset("nufft")


if __name__ == "__main__":
    unittest.main()
