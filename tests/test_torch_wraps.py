# %%
import unittest

import numpy as np
import torch
from aspire.image import Image
from aspire.nufft import anufft as aspire_anufft
from aspire.nufft import nufft as aspire_nufft
from aspire.operators import RadialCTFFilter
from aspire.source import Simulation
from aspire.utils import Rotation
from aspire.volume import LegacyVolume, rotated_grids
from torch.autograd import gradcheck

from solvar import nufft_plan, projection_funcs
from solvar.covar import Mean
from solvar.poses import PoseModule, rotvec_to_rotmat


class TestTorchWraps(unittest.TestCase):

    def setUp(self):
        self.img_size = 15
        self.num_imgs = 2048
        c = 5
        self.vols = (
            LegacyVolume(
                L=self.img_size,
                C=c,
                dtype=np.float32,
            ).generate()
            * 100
        )

        self.sim = Simulation(n=self.num_imgs, vols=self.vols, dtype=np.float32, amplitudes=1, offsets=0)

        rots = self.sim.rotations[:]
        self.pts_rot = rotated_grids(self.img_size, rots).reshape((3, len(rots), -1))

        self.device = torch.device("cuda:0")

    def test_centered_fft_pad_crop(self):
        us = 2
        vol_torch = torch.tensor(self.vols.asnumpy(), device=self.device)
        vol_torch_fft_padded = projection_funcs.centered_fft3(vol_torch, padding_size=(self.img_size * us,) * 3)
        vol_torch_ifft_cropped = projection_funcs.centered_ifft3(
            vol_torch_fft_padded, cropping_size=(self.img_size,) * 3
        ).real

        torch.testing.assert_close(vol_torch_ifft_cropped, vol_torch, rtol=1e-5, atol=1e-4)

    def test_nufft_forward(self):
        vols = self.vols
        pts_rot = self.pts_rot[:, 0][None, :]

        # singleton validation
        nufft_forward_aspire = aspire_nufft(vols[0].asnumpy(), pts_rot[0]).reshape(1, 1, self.img_size, -1)

        vol_torch = torch.tensor(vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch, plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        pts_rot_torch = (
            torch.remainder(torch.tensor(pts_rot.copy(), device=self.device) + torch.pi, 2 * torch.pi) - torch.pi
        )
        us = 2
        plan = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us, mode="bilinear")
        plan.setpts(pts_rot_torch)
        nufft_forward_disc = plan.execute_forward(
            projection_funcs.centered_fft3(vol_torch, padding_size=(self.img_size * us,) * 3)
        )
        nufft_forward_disc = nufft_forward_disc.cpu().numpy()

        threshold = np.mean(np.abs(nufft_forward_aspire))
        np.testing.assert_allclose(nufft_forward_torch, nufft_forward_aspire, rtol=1e-3, atol=threshold * 0.01)
        print(np.linalg.norm(nufft_forward_disc - nufft_forward_torch) / np.linalg.norm(nufft_forward_torch))
        np.testing.assert_array_less(
            np.linalg.norm(nufft_forward_disc - nufft_forward_torch) / np.linalg.norm(nufft_forward_torch), 0.2
        )

        # stack validation
        nufft_forward_aspire = aspire_nufft(vols, pts_rot[0]).reshape(vols.shape[0], 1, self.img_size, -1)

        vol_torch = torch.tensor(vols.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, vols.shape[0], device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        nufft_forward_torch = nufft_plan.nufft_forward(vol_torch, plan)
        nufft_forward_torch = nufft_forward_torch.cpu().numpy()

        pts_rot_torch = (
            torch.remainder(torch.tensor(pts_rot.copy(), device=self.device) + torch.pi, 2 * torch.pi) - torch.pi
        )
        plan = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us)
        plan.setpts(pts_rot_torch)
        nufft_forward_disc = plan.execute_forward(
            projection_funcs.centered_fft3(vol_torch, padding_size=(self.img_size * us,) * 3)
        )
        nufft_forward_disc = nufft_forward_disc.cpu().numpy()

        np.testing.assert_allclose(nufft_forward_torch, nufft_forward_aspire, rtol=1e-3, atol=threshold * 0.01)
        print(np.linalg.norm(nufft_forward_disc - nufft_forward_torch) / np.linalg.norm(nufft_forward_torch))
        np.testing.assert_array_less(
            np.linalg.norm(nufft_forward_disc - nufft_forward_torch) / np.linalg.norm(nufft_forward_torch), 0.2
        )

    def test_nufft_adjoint(self):
        # TODO : figure out why the difference between aspire's and the torch binding has rtol > 1e-4
        num_ims = 5

        # singleton validation
        pts_rot = self.pts_rot[:, 0][None, :]
        images = self.sim.images[0]
        from aspire.image import Image
        from aspire.numeric import fft, xp

        images = Image(xp.asnumpy(fft.centered_fft2(xp.asarray(images))))
        nufft_adjoint_aspire = aspire_anufft(images.asnumpy().reshape((1, -1)), pts_rot[0], (self.img_size,) * 3)
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real)) * 0.1

        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch, plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()[0]

        np.testing.assert_allclose(nufft_adjoint_torch, nufft_adjoint_aspire, rtol=1e-2, atol=threshold)

        # Testing with NufftPlanDiscretized
        us = 4
        pts_rot_torch = (
            torch.remainder(torch.tensor(pts_rot.copy(), device=self.device) + torch.pi, 2 * torch.pi) - torch.pi
        )
        plan_disc = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us, mode="bilinear")
        plan_disc.setpts(pts_rot_torch)
        nufft_adjoint_disc = plan_disc.execute_adjoint(im_torch)
        nufft_adjoint_disc = (
            projection_funcs.centered_ifft3(nufft_adjoint_disc, cropping_size=(self.img_size,) * 3).cpu().numpy()[0]
        )

        m = Mean(torch.tensor(nufft_adjoint_disc), 15, upsampling_factor=us)
        m.init_grid_correction("bilinear")

        np.testing.assert_allclose(
            nufft_adjoint_disc.real * (us * self.img_size) ** 3 / m.grid_correction.numpy(),
            nufft_adjoint_aspire.real,
            rtol=1e-2,
            atol=threshold * 3,
        )

        # Stack validation
        pts_rot = self.pts_rot[:, :num_ims]
        images = self.sim.images[:num_ims]
        images = Image(xp.asnumpy(fft.centered_fft2(xp.asarray(images))))
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real)) * 0.1
        nufft_adjoint_aspire = aspire_anufft(
            images.asnumpy().reshape((1, -1)), pts_rot.reshape(3, -1), (self.img_size,) * 3
        )
        threshold = np.mean(np.abs(nufft_adjoint_aspire.real)) * 0.1

        im_torch = torch.tensor(images.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device).transpose(0, 1))
        nufft_adjoint_torch = nufft_plan.nufft_adjoint(im_torch.reshape(num_ims, -1), plan)
        nufft_adjoint_torch = nufft_adjoint_torch.cpu().numpy()[0]

        np.testing.assert_allclose(nufft_adjoint_torch, nufft_adjoint_aspire, rtol=1e-2, atol=threshold)

        # Stack validation with NufftPlanDiscretized
        pts_rot_torch = (
            torch.remainder(torch.tensor(pts_rot.copy(), device=self.device).transpose(0, 1) + torch.pi, 2 * torch.pi)
            - torch.pi
        )
        plan_disc = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=us, mode="bilinear")
        plan_disc.setpts(pts_rot_torch)
        nufft_adjoint_disc = plan_disc.execute_adjoint(im_torch)
        nufft_adjoint_disc = (
            projection_funcs.centered_ifft3(nufft_adjoint_disc, cropping_size=(self.img_size,) * 3).cpu().numpy()[0]
        )

        np.testing.assert_allclose(
            nufft_adjoint_disc.real * (us * self.img_size) ** 3 / m.grid_correction.numpy(),
            nufft_adjoint_aspire.real,
            rtol=1e-2,
            atol=threshold * 3,
        )

    def test_grad_forward(self):
        pts_rot = np.float64(self.pts_rot[:, 0][None, :])
        vol = torch.randn((self.img_size,) * 3, dtype=torch.double, device=self.device) * 0.1
        vol.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, dtype=torch.float64, device=self.device)
        pts_rot = torch.tensor(pts_rot.copy(), device=self.device, requires_grad=True)
        pts_rot = (torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi).contiguous()
        plan.setpts(pts_rot)

        gradcheck(
            nufft_plan.TorchNufftForward.apply,
            (vol, pts_rot, plan, True),
            eps=1e-6,
            rtol=1e-4,
            atol=1e-3,
            nondet_tol=1e-5,
        )

    def test_grad_adjoint(self):
        pts_rot = np.float64(self.pts_rot[:, 0][None, :])
        im = torch.randn((self.img_size,) * 2, dtype=torch.double, device=self.device)
        im.requires_grad = True
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, dtype=torch.float64, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        gradcheck(nufft_plan.nufft_adjoint, (im, plan), eps=1e-6, rtol=1e-4, nondet_tol=1e-5)

    def test_vol_project(self):
        pts_rot = self.pts_rot[:, 0][None, :]

        vol_forward_aspire = self.sim.vol_forward(self.vols[0], 0, 1)

        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch, plan)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch, vol_forward_aspire, rtol=1e-3, atol=1e-3)

    def test_im_backproject(self):

        pts_rot = self.pts_rot[:, 0][None, :]
        imgs = self.sim.images[0]
        im_backproject_aspire = self.sim.im_backward(imgs, 0).asnumpy()

        im_torch = torch.tensor(imgs.asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        im_backproject_torch = projection_funcs.im_backward(im_torch, plan)
        im_backproject_torch = im_backproject_torch.cpu().numpy()

        np.testing.assert_allclose(im_backproject_torch, im_backproject_aspire, rtol=1e-3, atol=1e-3)

    def test_vol_project_ctf(self):
        sim = Simulation(
            n=1,
            vols=self.vols,
            dtype=np.float32,
            amplitudes=1,
            offsets=0,
            unique_filters=[RadialCTFFilter(defocus=1.5e4)],
        )
        rots = sim.rotations[:]
        pts_rot = rotated_grids(self.img_size, rots)
        pts_rot = pts_rot.reshape((3, -1))
        pts_rot = pts_rot[None, :]
        filter = torch.tensor(sim.unique_filters[0].evaluate_grid(self.img_size)).unsqueeze(0).to(self.device)
        vol_forward_aspire = sim.vol_forward(self.vols[0], 0, 1)

        vol_torch = torch.tensor(self.vols[0].asnumpy()).to(self.device)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device))
        vol_forward_torch = projection_funcs.vol_forward(vol_torch, plan, filter)
        vol_forward_torch = vol_forward_torch.cpu().numpy()

        np.testing.assert_allclose(vol_forward_torch, vol_forward_aspire, rtol=1e-3, atol=1e-3)

    def test_vol_project_fourier_slice(self):
        vol = torch.tensor(self.vols[0].asnumpy(), device=self.device)
        rot = np.array([np.eye(3)], self.vols.dtype)
        plan = nufft_plan.NufftPlan((self.img_size,) * 3, 1, device=self.device)
        plan.setpts(torch.tensor(rotated_grids(self.img_size, rot).copy(), device=self.device).reshape((1, 3, -1)))

        vol_forward = projection_funcs.vol_forward(vol, plan)
        vol_forward_fourier = projection_funcs.centered_fft2(vol_forward)[0]

        vol_fourier = projection_funcs.centered_fft3(vol)
        vol_fourier_slice = vol_fourier[0][self.img_size // 2]

        torch.testing.assert_close(vol_fourier_slice, vol_forward_fourier * self.img_size, rtol=5e-3, atol=5e-3)

    def test_batch_nufft_grad(self):
        batch_size = 8
        pts_rot = self.pts_rot[:, :batch_size]
        pts_rot = torch.tensor(pts_rot.copy(), device=self.device).transpose(0, 1)
        vol_torch = torch.tensor(self.vols.asnumpy(), device=self.device, requires_grad=True)
        num_vols = vol_torch.shape[0]
        plans = [nufft_plan.NufftPlan((self.img_size,) * 3, num_vols, device=self.device) for i in range(batch_size)]
        for i in range(batch_size):
            plans[i].setpts(pts_rot[i].unsqueeze(0))
        vol_forward = torch.zeros(
            (batch_size, num_vols, self.img_size, self.img_size), dtype=vol_torch.dtype, device=self.device
        )
        for i in range(batch_size):
            vol_forward[i] = projection_funcs.vol_forward(vol_torch, plans[i])

        v1 = torch.norm(vol_forward)
        v1.backward()
        vol_forward_grad = vol_torch.grad

        vol_torch = torch.tensor(self.vols.asnumpy(), device=self.device, requires_grad=True)
        batch_plans = nufft_plan.NufftPlan((self.img_size,) * 3, num_vols, device=self.device)
        batch_plans.setpts(pts_rot)
        batch_vol_forward = projection_funcs.vol_forward(vol_torch, batch_plans)

        v2 = torch.norm(batch_vol_forward)
        v2.backward()
        batch_vol_forward_grad = vol_torch.grad

        torch.testing.assert_close(vol_forward, batch_vol_forward, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(vol_forward_grad, batch_vol_forward_grad, rtol=5e-3, atol=5e-3)

    def test_rotmat_rotvec(self):
        rotations = self.sim.rotations  # (N, 3, 3)
        rotvecs = torch.tensor(Rotation.from_matrix(rotations).as_rotvec(), dtype=torch.float32)

        romats = rotvec_to_rotmat(rotvecs)
        # Aspire implementation
        aspire_rotmats = torch.tensor(Rotation.from_rotvec(rotvecs.numpy()).matrices, dtype=torch.float32)
        torch.testing.assert_close(romats, aspire_rotmats, rtol=1e-5, atol=1e-5)

    def test_pose_module_rots(self):
        rotations = self.sim.rotations
        init_rotvec = torch.tensor(Rotation.from_matrix(rotations).as_rotvec(), dtype=torch.float32)
        pose_module = PoseModule(init_rotvec, torch.zeros(len(init_rotvec), 2), self.img_size)
        index = torch.tensor([5, 13, 192, 153])
        pts_rot = torch.tensor(self.pts_rot.copy()).reshape(3, -1, self.img_size**2)[:, index].transpose(0, 1)
        pts_rot = (
            torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi
        )  # After rotating the grids some of the points can be outside the [-pi , pi]^3 cube
        module_pts_rot, _ = pose_module(index)
        torch.testing.assert_close(pts_rot, module_pts_rot, rtol=1e-3, atol=1e-3)

    def test_pose_module_offsets(self):
        N = 100
        offsets = torch.randn((N, 2), dtype=torch.float32) * 5
        init_rotvec = torch.tensor(Rotation.from_matrix(self.sim.rotations[:N]).as_rotvec(), dtype=torch.float32)
        pose_module = PoseModule(init_rotvec, offsets, self.img_size)

        images = torch.randn((N, self.img_size, self.img_size), dtype=torch.float32)
        _, phase_shift = pose_module(torch.arange(N))
        module_shifted_images = projection_funcs.centered_ifft2(
            projection_funcs.centered_fft2(images) * phase_shift
        ).real

        aspire_shifted_images = Image(images.numpy()).shift(-offsets).asnumpy()

        torch.testing.assert_close(module_shifted_images, torch.tensor(aspire_shifted_images), rtol=1e-3, atol=1e-3)

    def test_downsample_nufft(self):
        rotations = self.sim.rotations
        init_rotvec = torch.tensor(Rotation.from_matrix(rotations).as_rotvec(), dtype=torch.float32)
        pose_module = PoseModule(init_rotvec, torch.zeros(len(init_rotvec), 2), self.img_size)
        index = torch.tensor([5, 13, 192, 153])

        volume = torch.randn((self.img_size,) * 3, dtype=torch.float32)
        nufft_plans = nufft_plan.NufftPlanDiscretized(volume.shape, upsample_factor=2, mode="bilinear")

        volume = projection_funcs.centered_fft3(volume, padding_size=(self.img_size * 2,) * 3).unsqueeze(0)
        nufft_plans.setpts(pose_module(index)[0])
        v1 = nufft_plans.execute_forward(volume)

        v1_forward = projection_funcs.vol_forward(volume, nufft_plans, fourier_domain=True)

        nufft_plans.setpts(pose_module(index, ds_resolution=self.img_size // 2)[0])
        v2 = nufft_plans.execute_forward(volume)
        v2_forward = projection_funcs.vol_forward(volume, nufft_plans, fourier_domain=True)

        v1 = projection_funcs.crop_image(v1, self.img_size // 2)
        print(v1_forward.shape)
        v1_forward = projection_funcs.crop_image(v1_forward, self.img_size // 2)

        torch.testing.assert_close(v1, v2, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(v1_forward, v2_forward, rtol=5e-3, atol=5e-3)

    def test_vol_forward_im_backward_adjointness(self):
        """Test that vol_forward and im_backward are adjoint operators.

        For adjoint operators A and A*, we should have: <A(v), i> = <v, A*(i)> where <.,.> denotes
        the inner product.
        """
        batch_size = 32
        pts_rot = self.pts_rot[:, :batch_size]
        dtype = torch.float64
        upsample_factor = 2

        # Create random test data
        rand_tensor = lambda sz: torch.randn(sz, dtype=dtype, device=self.device)
        volume = torch.complex(
            rand_tensor((self.img_size * upsample_factor,) * 3), rand_tensor((self.img_size * upsample_factor,) * 3)
        )
        image = torch.complex(*((rand_tensor((batch_size, self.img_size, self.img_size)),) * 2))

        # Set up NUFFT plan
        # plan = nufft_plan.NufftPlan((self.img_size,)*3, 1, device=self.device,dtype=dtype)
        plan = nufft_plan.NufftPlanDiscretized((self.img_size,) * 3, upsample_factor=upsample_factor, mode="nearest")
        plan.setpts(torch.tensor(pts_rot.copy(), device=self.device, dtype=dtype))

        # Compute vol_forward(volume) and im_backward(image)
        vol_forward_result = projection_funcs.vol_forward(volume.unsqueeze(0), plan, fourier_domain=True).squeeze()
        im_backward_result = projection_funcs.im_backward(image, plan, fourier_domain=True).squeeze()

        # Compute inner products
        # <vol_forward(volume), image>
        inner_product_1 = torch.sum(vol_forward_result * torch.conj(image))

        # <volume, im_backward(image)>
        inner_product_2 = torch.sum(volume * torch.conj(im_backward_result))

        # Check adjointness: these should be equal
        torch.testing.assert_close(inner_product_1, inner_product_2, rtol=1e-8, atol=1e-4)

        # Also test with CTF filters
        filter_tensor = torch.randn((batch_size,) + (self.img_size,) * 2, dtype=dtype, device=self.device)

        vol_forward_ctf = projection_funcs.vol_forward(
            volume.unsqueeze(0), plan, fourier_domain=True, filters=filter_tensor
        ).squeeze()
        im_backward_ctf = projection_funcs.im_backward(
            image, plan, fourier_domain=True, filters=filter_tensor
        ).squeeze()

        inner_product_1_ctf = torch.sum(vol_forward_ctf * torch.conj(image))
        inner_product_2_ctf = torch.sum(volume * torch.conj(im_backward_ctf))

        torch.testing.assert_close(inner_product_1_ctf, inner_product_2_ctf, rtol=1e-8, atol=1e-4)


if __name__ == "__main__":

    unittest.main()
