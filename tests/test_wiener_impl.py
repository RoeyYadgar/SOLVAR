import time
import unittest

import numpy as np
import torch
from aspire.denoising import src_wiener_coords
from aspire.operators import RadialCTFFilter
from aspire.source import Simulation
from aspire.volume import Volume

from cov3d.covar import Covar
from cov3d.dataset import CovarDataset
from cov3d.nufft_plan import NufftPlanDiscretized, NufftSpec
from cov3d.utils import volsCovarEigenvec
from cov3d.wiener_coords import latentMAP, wiener_coords

from .utils import process_volume


class TestWienerImpl(unittest.TestCase):

    def test_wiener_impl(self):
        L = 64
        r = 2

        idx = np.linspace(0, 99, r + 1).astype(int)
        vols = [process_volume(Volume.load(f"/home/ry295/pi_data/igg_1d/vols/128_org/{i:03d}.mrc"), L) for i in idx]
        vols = Volume(np.concatenate(vols))

        source = Simulation(
            n=25000,
            vols=vols,
            dtype=np.float32,
            amplitudes=1,
            offsets=0,
            unique_filters=[RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
        )
        mean_vol = Volume(np.mean(vols, axis=0))
        noise_var = 1
        dataset = CovarDataset(source, noise_var, mean_volume=mean_vol, mask=None)
        device = torch.device("cuda:0")

        cov = Covar(L, r, vectors=torch.tensor(volsCovarEigenvec(vols)).reshape(r, L, L, L)).to(device)
        eigenvecs, eigenvals = cov.eigenvecs

        t = time.time()
        coords = wiener_coords(dataset, eigenvecs, eigenvals)
        elapsed_time_impl = time.time() - t

        t = time.time()
        coords2 = latentMAP(dataset, eigenvecs, eigenvals)
        elapsed_time_implMap = time.time() - t

        nufft_spec = NufftSpec(nufft_type=NufftPlanDiscretized, upsample_factor=2, mode="bilinear")
        t = time.time()
        coords3 = latentMAP(dataset, eigenvecs, eigenvals, nufft_spec=nufft_spec)
        elapsed_time_implMapNN = time.time() - t

        eigenvecs = Volume(eigenvecs.cpu().numpy())
        eigenvals = np.diag(eigenvals.cpu().numpy())
        t = time.time()
        coords_aspire = src_wiener_coords(source, mean_vol, eigenvecs, eigenvals, noise_var=noise_var)
        elapsed_time_aspire = time.time() - t

        np.testing.assert_allclose(coords.cpu().numpy(), coords_aspire.T, rtol=1e-6, atol=1e-2)
        torch.testing.assert_close(coords, coords2, rtol=1e-6, atol=1e-2)
        torch.testing.assert_close(coords2, coords3, rtol=1e-6, atol=1e-2)
        print(f"Elapsed time of aspire implementation : {elapsed_time_aspire}")
        print(f"Elapsed time of own implementation : {elapsed_time_impl}")
        print(f"Elapsed time of MAP implementation : {elapsed_time_implMap}")
        print(f"Elapsed time of MAP implementation, Trilinear interp: {elapsed_time_implMapNN}")


if __name__ == "__main__":
    unittest.main()
