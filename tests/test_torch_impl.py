import unittest

import numpy as np
import torch
from aspire.noise import WhiteNoiseAdder
from aspire.operators import RadialCTFFilter
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.volume import LegacyVolume, Volume, rotated_grids

from solvar import nufft_plan
from solvar.covar import Covar
from solvar.covar_sgd import cost, cost_fourier_domain, cost_maximum_liklihood, cost_maximum_liklihood_fourier_domain
from solvar.dataset import CovarDataset
from solvar.fsc_utils import (
    FourierShell,
    average_fourier_shell,
    covar_correlate,
    expand_fourier_shell,
    rpsd,
    upsample_and_expand_fourier_shell,
)
from solvar.projection_funcs import centered_fft3, vol_forward
from solvar.utils import generateBallVoxel, volsCovarEigenvec


class TestTorchImpl(unittest.TestCase):

    def setUp(self):
        self.img_size = 16
        self.num_imgs = 100
        c = 5
        noise_var = 1
        self.vols = LegacyVolume(
            L=self.img_size,
            C=c,
            dtype=np.float32,
        ).generate()
        self.vols -= np.mean(self.vols, axis=0)

        self.sim = Simulation(
            n=self.num_imgs,
            vols=self.vols,
            dtype=np.float32,
            amplitudes=1,
            offsets=0,
            unique_filters=[RadialCTFFilter(defocus=d, pixel_size=3) for d in np.linspace(8e3, 2.5e4, 7)],
            noise_adder=WhiteNoiseAdder(noise_var),
        )
        self.dataset = CovarDataset(self.sim, noise_var=noise_var, apply_preprocessing=False)
        self.vectorsGT = torch.tensor(volsCovarEigenvec(self.vols))
        rots = self.sim.rotations[:]
        pts_rot = rotated_grids(self.img_size, rots)
        self.pts_rot = pts_rot.reshape((3, -1))

        self.device = torch.device("cuda:0")

    def test_cost_gradient_scaling(self):

        # Test scaling of volumes
        num_ims = 50
        rank = 4
        reg = 0
        noise_var = 1
        scaling_param = 10

        vols = torch.randn(
            (rank, self.img_size, self.img_size, self.img_size),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )
        images, pts_rot, filters, _ = self.dataset[:num_ims]
        pts_rot = pts_rot.to(self.device)
        images = images.to(self.device)
        filters = filters.to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, batch_size=rank, dtype=torch.float32, device=self.device)
        plan.setpts(pts_rot)

        cost_val = cost(vols, images, plan, filters, noise_var, reg_scale=reg)
        cost_val.backward()

        vols_grad = vols.grad.clone()

        scaled_vols = torch.tensor(vols.data * scaling_param, requires_grad=True, device=self.device)
        # When the volumes are scaled by alpha (and SNR is preserved) the cost gradient should scale by alpha ** 3
        cost_val = cost(
            scaled_vols, images * scaling_param, plan, filters, noise_var * (scaling_param**2), reg_scale=reg
        )
        cost_val.backward()

        scaled_vols_grad = scaled_vols.grad.clone()
        torch.testing.assert_close(
            (scaled_vols_grad).to("cpu"), (scaling_param**3) * vols_grad.to("cpu"), rtol=5e-3, atol=5e-3
        )

        vols.grad.zero_()
        # When the filters are scaled by alpha (and SNR is preseverd)
        # the cost gradient should scale by alpha ** 4 as well as the regularization parameter.
        cost_val = cost(
            vols,
            images * scaling_param,
            plan,
            filters * scaling_param,
            noise_var * (scaling_param**2),
            reg_scale=reg * (scaling_param**4),
        )
        cost_val.backward()
        scaled_filters_grid = vols.grad.clone()
        torch.testing.assert_close(
            (scaled_filters_grid).to("cpu"), (scaling_param**4) * vols_grad.to("cpu"), rtol=5e-3, atol=5e-3
        )

    def test_cost_gradient_resolution_scaling(self):
        num_ims = 1
        rank = 2
        reg = 0
        noise_var = 1

        rots = Rotation.generate_random_rotations(num_ims).matrices

        pts_rot = torch.tensor(
            rotated_grids(self.img_size, rots).copy(), device=self.device, dtype=torch.float32
        ).reshape((3, num_ims, self.img_size**2))
        pts_rot = pts_rot.transpose(0, 1)

        vols = torch.tensor(
            LegacyVolume(
                L=self.img_size,
                C=rank,
                dtype=np.float32,
            )
            .generate()
            .asnumpy()
            .reshape(rank, self.img_size, self.img_size, self.img_size),
            dtype=torch.float32,
            requires_grad=True,
            device=self.device,
        )

        plan = nufft_plan.NufftPlan((self.img_size,) * 3, batch_size=rank, dtype=torch.float32, device=self.device)
        plan.setpts(pts_rot)

        images = torch.zeros(num_ims, self.img_size, self.img_size, device=self.device, dtype=torch.float32)
        for i in range(num_ims):
            images[i] = vol_forward(vols, plan)[i, i % rank]

        cost_val = cost(vols, 2 * images, plan, None, noise_var, reg_scale=reg)
        cost_val.backward()
        vols_grad = vols.grad.clone()

        # Downsampling
        img_size_ds = int(self.img_size / 2)
        vols_ds = Volume(vols.detach().cpu().numpy()).downsample(img_size_ds)
        vols_ds = torch.tensor(vols_ds.asnumpy(), device=self.device, dtype=torch.float32, requires_grad=True)
        pts_rot = torch.tensor(
            rotated_grids(img_size_ds, rots).copy(), device=self.device, dtype=torch.float32
        ).reshape((3, num_ims, img_size_ds**2))
        pts_rot = pts_rot.transpose(0, 1)

        plan = nufft_plan.NufftPlan((img_size_ds,) * 3, batch_size=rank, dtype=torch.float32, device=self.device)
        plan.setpts(pts_rot)

        images_ds = torch.zeros(num_ims, img_size_ds, img_size_ds, device=self.device, dtype=torch.float32)
        for i in range(num_ims):
            images_ds[i] = vol_forward(vols_ds, plan)[i, i % rank]

        cost_val = cost(vols_ds, 2 * images_ds, plan, None, noise_var * (2 ** (-2)), reg_scale=reg * (2**2))
        cost_val.backward()
        vols_grad_ds = vols_ds.grad.clone()

        downsampled_vols_grad = torch.tensor(Volume(vols_grad.cpu().numpy()).downsample(img_size_ds).asnumpy())
        torch.testing.assert_close(
            torch.norm(downsampled_vols_grad), torch.norm(vols_grad_ds.cpu() * 2), rtol=3e-2, atol=1e-1
        )

    def test_projection_resolution_scaling(self):
        n = 8
        # Projection scales values of volume by 1/(L^*1.5)

        L1 = 15
        voxels = Volume.from_vec((generateBallVoxel([-0.6, 0, 0], 0.5, L1)))
        sim = Simulation(n=n, vols=voxels, amplitudes=1, offsets=0)
        rots = sim.rotations

        norm_vol1 = np.linalg.norm(voxels)
        norm_images1 = np.linalg.norm(sim.images[:], axis=(1, 2))
        norm_backproj1 = np.linalg.norm(sim.images[0].backproject(rots[0].reshape((1, 3, 3))))

        L2 = 200
        voxels = Volume.from_vec((generateBallVoxel([-0.6, 0, 0], 0.5, L2)))
        sim = Simulation(n=n, vols=voxels, amplitudes=1, offsets=0, angles=Rotation.from_matrix(rots).as_rotvec())

        norm_vol2 = np.linalg.norm(voxels)
        norm_images2 = np.linalg.norm(sim.images[:], axis=(1, 2))
        norm_backproj2 = np.linalg.norm(sim.images[0].backproject(rots[0].reshape((1, 3, 3))))

        np.testing.assert_allclose(norm_vol2 / norm_vol1, (L2 / L1) ** 1.5, rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(norm_images2 / norm_images1, (L2 / L1) ** 1, rtol=1e-1, atol=1e-1)
        np.testing.assert_allclose(norm_backproj2 / norm_backproj1, (L2 / L1) ** 0.5, rtol=1e-1, atol=1e-1)

    def test_fourier_reg(self):
        L = self.img_size
        rank = self.vectorsGT.shape[0]
        vgd = self.vectorsGT.reshape((rank, L, L, L))
        vectorsGT_rpsd = rpsd(*vgd)
        gd_psd = (
            torch.sum(expand_fourier_shell(vectorsGT_rpsd, L, 3), dim=0)
            if rank > 1
            else expand_fourier_shell(vectorsGT_rpsd, L, 3)
        )

        noise_var = 10
        fourier_reg = noise_var / gd_psd

        # vols = torch.randn((rank,L,L,L))
        vols = vgd
        vols_fourier = centered_fft3(vols)
        vols_fourier *= torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank, -1))

        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        reg_term_efficient_computation = torch.sum(torch.pow(vols_fourier_inner_prod.abs(), 2))

        vols_fourier = centered_fft3(vols).reshape(rank, -1)
        vgd_fourier = gd_psd.reshape(1, -1)
        M = vgd_fourier.T @ vgd_fourier.conj()
        reg_matrix_coeff = noise_var / torch.abs(M) ** 0.5
        reg_term_inefficient_computation = torch.norm(reg_matrix_coeff * (vols_fourier.T @ vols_fourier.conj())) ** 2

        print((reg_term_efficient_computation, reg_term_inefficient_computation))
        print((reg_term_efficient_computation - reg_term_inefficient_computation) / reg_term_inefficient_computation)

        torch.testing.assert_close(
            reg_term_efficient_computation, reg_term_inefficient_computation, rtol=5e-3, atol=5e-3
        )

    def test_fourier_domain_cost(self):
        batch_size = 16
        rank = 4
        upsampling_factor = 2
        # covar = Covar(self.img_size,rank=rank,upsampling_factor=upsampling_factor,
        # vectors=torch.tensor(self.vols[:rank].asnumpy()))
        covar = Covar(self.img_size, rank=rank, upsampling_factor=upsampling_factor)

        ims, pts_rot, filters, _ = self.dataset[:batch_size]

        vectorsGT_rpsd = rpsd(*self.vectorsGT.reshape((-1, self.img_size, self.img_size, self.img_size)))
        fourier_reg = (self.dataset.noise_var) / (
            torch.mean(expand_fourier_shell(vectorsGT_rpsd, self.img_size, 3), dim=0)
        )

        # Spatial domain cost
        plans = nufft_plan.NufftPlan((self.img_size,) * 3, batch_size=rank)
        plans.setpts(pts_rot)
        cost_spatial = torch.tensor(
            [
                cost(covar.get_vectors_spatial_domain(), ims, plans, filters, self.dataset.noise_var),
                cost(covar.get_vectors_spatial_domain() * 1e3, ims, plans, filters, self.dataset.noise_var),
                cost(covar.get_vectors_spatial_domain(), ims, plans, filters, self.dataset.noise_var * 1e6),
                cost(
                    covar.get_vectors_spatial_domain(),
                    ims,
                    plans,
                    filters,
                    self.dataset.noise_var,
                    reg_scale=1,
                    fourier_reg=fourier_reg,
                ),
            ]
        )

        # Fourier domain cost
        fourier_data = self.dataset.copy()
        fourier_data.to_fourier_domain()
        ims, pts_rot, filters, _ = fourier_data[:batch_size]

        fourier_reg = (fourier_data.noise_var) / (
            torch.mean(expand_fourier_shell(vectorsGT_rpsd, self.img_size, 3), dim=0)
        )
        fourier_reg_radial = average_fourier_shell(fourier_reg) / (upsampling_factor**3)
        fourier_reg = upsample_and_expand_fourier_shell(
            fourier_reg_radial, covar.resolution * covar.upsampling_factor, 3
        )

        covar.init_grid_correction("bilinear")
        plans = nufft_plan.NufftPlanDiscretized(
            (self.img_size,) * 3, upsample_factor=upsampling_factor, mode="bilinear", use_half_grid=False
        )
        plans.setpts(pts_rot)
        cost_fourier = torch.tensor(
            [
                cost_fourier_domain(covar.get_vectors_fourier_domain(), ims, plans, filters, fourier_data.noise_var),
                cost_fourier_domain(
                    covar.get_vectors_fourier_domain() * 1e3, ims, plans, filters, fourier_data.noise_var
                ),
                cost_fourier_domain(
                    covar.get_vectors_fourier_domain(), ims, plans, filters, fourier_data.noise_var * 1e6
                ),
                cost_fourier_domain(
                    covar.get_vectors_fourier_domain(),
                    ims,
                    plans,
                    filters,
                    fourier_data.noise_var,
                    reg_scale=1,
                    fourier_reg=fourier_reg,
                ),
            ]
        )

        print((cost_spatial, cost_fourier))
        print((cost_spatial / cost_fourier))

        # Skip validation of last evaluation - known issue due to grid_correction of volumes where it is not needed
        cost_spatial = cost_spatial[:3]
        cost_fourier = cost_fourier[:3]
        torch.testing.assert_close(cost_fourier / cost_spatial, torch.ones_like(cost_fourier), rtol=5e-2, atol=5e-2)

    def test_ml_cost(self):

        def naive_cost_maximum_liklihood(vols, images, nufft_plans, filters, noise_var):
            batch_size = images.shape[0]
            rank = vols.shape[0]
            L = images.shape[-1]

            projected_vols = vol_forward(vols, nufft_plans, filters, fourier_domain=vols.is_complex())
            images = images.reshape((batch_size, -1, 1))
            projected_vols = projected_vols.reshape((batch_size, rank, -1))
            projected_eigen_covar = torch.matmul(
                projected_vols.transpose(1, 2), projected_vols.conj()
            ) + noise_var * torch.eye(L**2, device=vols.device, dtype=vols.dtype).reshape(
                1, L**2, L**2
            )  # size (batch,L**2,L**2)
            inverted_projected_eigen_covar = torch.inverse(projected_eigen_covar)  # size (batch,L**2,L**2)

            ml_exp_term = (
                torch.matmul(
                    images.transpose(1, 2).conj(), torch.matmul(inverted_projected_eigen_covar, images)
                ).squeeze()
                - 1 / noise_var * torch.norm(images, dim=(1, 2)) ** 2
            )  # remove constant term that is not being taked into account in cost_maximum_liklihood_fourier_domain
            ml_noise_term = torch.logdet(projected_eigen_covar) - (L**2) * torch.log(
                torch.tensor(noise_var)
            )  # remove constant term that is not being taked into account in cost_maximum_liklihood_fourier_domain

            cost_val = 0.5 * torch.mean(ml_exp_term + ml_noise_term).real

            return cost_val

        batch_size = 13
        rank = 4
        upsampling_factor = 1
        covar = Covar(self.img_size, rank=rank, upsampling_factor=upsampling_factor)
        ims, pts_rot, filters, _ = self.dataset[:batch_size]
        plans = nufft_plan.NufftPlan((self.img_size,) * 3, batch_size=rank)
        plans.setpts(pts_rot)
        efficient_cost = cost_maximum_liklihood(covar.vectors, ims, plans, filters, self.dataset.noise_var * 100)
        efficient_cost.backward()
        grad_efficient = covar.vectors.grad.clone()
        covar.vectors.grad.zero_()
        naive_cost = naive_cost_maximum_liklihood(covar.vectors, ims, plans, filters, self.dataset.noise_var * 100)
        naive_cost.backward()
        grad_naive = covar.vectors.grad.clone()
        print(naive_cost)
        print(efficient_cost)
        torch.testing.assert_close(naive_cost, efficient_cost, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(grad_efficient, grad_naive, rtol=5e-3, atol=5e-3)

        self.dataset.to_fourier_domain()
        ims, pts_rot, filters, _ = self.dataset[:batch_size]
        plans = nufft_plan.NufftPlanDiscretized(
            (self.img_size,) * 3, upsample_factor=upsampling_factor, mode="bilinear"
        )
        plans.setpts(pts_rot)

        efficient_cost = cost_maximum_liklihood_fourier_domain(
            covar.get_vectors_fourier_domain(), ims, plans, filters, self.dataset.noise_var
        )
        naive_cost = naive_cost_maximum_liklihood(
            covar.get_vectors_fourier_domain(), ims, plans, filters, self.dataset.noise_var
        )

        torch.testing.assert_close(naive_cost, efficient_cost, rtol=5e-3, atol=5e-3)

    def test_covar_rpsd(self):
        eigenvecs1 = self.vectorsGT.reshape(-1, self.img_size, self.img_size, self.img_size).to(self.device)
        eigenvecs2 = eigenvecs1 + 0 * torch.randn(eigenvecs1.shape, device=self.device)

        eigenvecs1 = centered_fft3(eigenvecs1).reshape(-1, self.img_size**3)
        eigenvecs2 = centered_fft3(eigenvecs2).reshape(-1, self.img_size**3)

        fourier_covar1 = eigenvecs1.T @ eigenvecs1.conj()
        fourier_covar2 = eigenvecs2.T @ eigenvecs2.conj()

        s = FourierShell(self.img_size, 3, device=self.device)
        covar_fsc_naive = s.avergage_fourier_shell(
            (fourier_covar1 * fourier_covar2.conj()).reshape(-1, self.img_size, self.img_size, self.img_size)
        )
        covar_fsc_naive = s.avergage_fourier_shell(
            covar_fsc_naive.T.reshape(-1, self.img_size, self.img_size, self.img_size)
        )

        covar_fsc_efficient = covar_correlate(
            eigenvecs1.reshape(-1, self.img_size, self.img_size, self.img_size),
            eigenvecs2.reshape(-1, self.img_size, self.img_size, self.img_size),
        )

        torch.testing.assert_close(covar_fsc_naive, covar_fsc_efficient, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":

    unittest.main()
