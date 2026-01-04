import copy
import logging
import random
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from aspire.source import ImageSource as ASPIREImageSource
from aspire.utils import Rotation, support_mask
from aspire.volume import Volume, rotated_grids
from torch.utils.data import Dataset
from tqdm import tqdm

from cov3d.covar import Mean
from cov3d.fsc_utils import average_fourier_shell, sum_over_shell
from cov3d.nufft_plan import NufftPlan, NufftPlanDiscretized
from cov3d.poses import PoseModule, rotvec_to_rotmat
from cov3d.projection_funcs import (
    centered_fft2,
    centered_ifft2,
    get_mask_threshold,
    im_backward,
    preprocess_image_batch,
)
from cov3d.source import ImageSource, SimulatedSource
from cov3d.utils import get_torch_device, set_module_grad, soft_edged_kernel

logger = logging.getLogger(__name__)


class CovarDataset(Dataset):
    """Dataset for covariance estimation with preprocessing and pose handling.

    Attributes:
        resolution: Image resolution
        rot_vecs: Rotation vectors for each particle
        offsets: Translation offsets for each particle
        images: Particle images
        filters: CTF filters for each particle
        noise_var: Noise variance estimate
        data_inverted: Whether data has been sign-inverted
        _in_spatial_domain: Whether data is currently in spatial domain
        dtype: Data type for tensors
        mask: Volume mask for masking operations
    """

    def __init__(
        self,
        src: Union[ImageSource, ASPIREImageSource, SimulatedSource],
        noise_var: float,
        mean_volume: Optional[Volume] = None,
        mask: Optional[Volume] = None,
        invert_data: bool = False,
        apply_preprocessing: bool = True,
    ) -> None:
        if isinstance(src, ImageSource):
            self._init_from_source(src)
        else:
            self._init_from_aspire_source(src)
        self.pts_rot = self.compute_pts_rot(self.rot_vecs)
        self.noise_var = noise_var
        self.data_inverted = invert_data
        self._in_spatial_domain = True

        if self.data_inverted:
            self.images = -1 * self.images

        if apply_preprocessing:
            self.preprocess_from_modules(
                *self.construct_mean_pose_modules(mean_volume, mask, self.rot_vecs, self.offsets)
            )

        self.dtype = self.images.dtype
        self.mask = torch.tensor(mask.asnumpy()) if mask is not None else None
        self.contrasts = torch.ones((len(self)), dtype=self.offsets.dtype)

        self.radial_filters_gain = None
        self.signal_rpsd = None
        self.signal_var = None

    def _init_from_source(self, source: "ImageSource") -> None:
        """Initialize dataset from ImageSource object.

        Args:
            source: ImageSource object containing particle data
        """
        self.resolution = source.resolution
        # TODO: replace with non ASPIRE implemntation?
        self.rot_vecs = torch.tensor(Rotation(source.rotations.numpy()).as_rotvec(), dtype=source.rotations.dtype)
        self.offsets = source.offsets

        batch_size = 1024 * 16
        images_list = []
        ctf_list = []
        for i in range(0, len(source), batch_size):
            batch_idx = torch.arange(i, min(i + batch_size, len(source)))

            images_batch = source.images(batch_idx).cpu()
            images_list.append(images_batch)

            ctf_batch = source.get_ctf(batch_idx).cpu()
            ctf_list.append(ctf_batch)
        self.images = torch.cat(images_list, dim=0)
        self.filters = torch.cat(ctf_list, dim=0)

    def _init_from_aspire_source(self, source: Union[ASPIREImageSource, SimulatedSource]) -> None:
        """Initialize dataset from ASPIRE source object.

        Args:
            source: ASPIRE source object containing particle data
        """
        self.resolution = source.L
        self.rot_vecs = torch.tensor(Rotation(source.rotations).as_rotvec().astype(source.rotations.dtype))
        self.offsets = torch.tensor(source.offsets, dtype=self.rot_vecs.dtype)
        self.images = torch.tensor(source.images[:].asnumpy())

        filter_indices = torch.tensor(
            source.filter_indices.astype(int)
        )  # For some reason ASPIRE store filter_indices as string for some star files
        num_filters = len(source.unique_filters)
        unique_filters = torch.zeros((num_filters, source.L, source.L), dtype=self.images.dtype)
        for i in range(num_filters):
            unique_filters[i] = torch.tensor(source.unique_filters[i].evaluate_grid(source.L))

        self.filters = unique_filters[filter_indices]

    def __len__(self) -> int:
        """Return the number of particles in the dataset.

        Returns:
            Number of particles
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Get a particle and its associated data by index.

        Args:
            idx: Particle index

        Returns:
            Tuple containing (image, rotation_points, filter, index)
        """
        return self.images[idx], self.pts_rot[idx], self.filters[idx], idx

    def compute_pts_rot(self, rotvecs: torch.Tensor) -> torch.Tensor:
        """Compute rotated grid points for given rotation vectors.

        Args:
            rotvecs: Rotation vectors of shape (N, 3)

        Returns:
            Rotated grid points of shape (N, 3, resolution^2)
        """
        rotations = Rotation.from_rotvec(rotvecs.numpy())
        pts_rot = torch.tensor(rotated_grids(self.resolution, rotations.matrices).copy()).reshape(
            (3, rotvecs.shape[0], self.resolution**2)
        )  # TODO : replace this with torch affine_grid with size (N,1,L,L,1)
        pts_rot = pts_rot.transpose(0, 1)
        pts_rot = (
            torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi
        )  # After rotating the grids some of the points can be outside the [-pi , pi]^3 cube

        return pts_rot

    def construct_mean_pose_modules(
        self,
        mean_volume: Optional[Volume],
        mask: Optional[Volume],
        rot_vecs: torch.Tensor,
        offsets: torch.Tensor,
        fourier_domain: bool = False,
    ) -> List[torch.nn.Module]:
        """Construct mean and pose modules used for preprocessing (Substraction of projected mean
        volume)

        Args:
            mean_volume: Mean volume for projection
            mask: Volume mask for masking operations
            rot_vecs: Rotation vectors for each particle
            offsets: Translation offsets for each particle
            fourier_domain: Whether to work in Fourier domain

        Returns:
            List of PyTorch modules for preprocessing
        """
        L = self.resolution
        device = get_torch_device()

        if isinstance(mask, Volume):
            mask = torch.tensor(mask.asnumpy())

        if mean_volume is not None:
            mean = Mean(
                torch.tensor(mean_volume.asnumpy()),
                L,
                fourier_domain=fourier_domain,
                volume_mask=mask if mask is not None else None,
            )
            if fourier_domain:
                mean.init_grid_correction("bilinear")

            dtype = mean.dtype
            upsampling_factor = mean.upsampling_factor
        else:
            mean = None
            dtype = torch.float32
            upsampling_factor = 2
        pose = PoseModule(rot_vecs, offsets, L)
        nufft_plan = (
            NufftPlan((self.resolution,) * 3, batch_size=1, dtype=dtype, device=device)
            if not fourier_domain
            else NufftPlanDiscretized((self.resolution,) * 3, upsample_factor=upsampling_factor, mode="bilinear")
        )

        return mean, pose, nufft_plan

    def preprocess_from_modules(
        self,
        mean_module: torch.nn.Module,
        pose_module: torch.nn.Module,
        nufft_plan: Optional[Union[NufftPlan, NufftPlanDiscretized]] = None,
        batch_size: int = 1024,
    ) -> None:
        """Preprocess dataset using mean and pose modules.

        Args:
            mean_module: Module for mean volume projection
            pose_module: Module for pose optimization
            nufft_plan: NUFFT plan for projection operations
            batch_size: Batch size for processing
        """
        device = get_torch_device()
        mean_module = mean_module.to(device)
        pose_module = pose_module.to(device)
        if nufft_plan is None:
            nufft_plan = (
                NufftPlan((self.resolution,) * 3, batch_size=1, dtype=mean_module.dtype, device=device)
                if mean_module._in_spatial_domain
                else NufftPlanDiscretized(
                    (self.resolution,) * 3, upsample_factor=mean_module.upsampling_factor, mode="bilinear"
                )
            )

        softening_kernel_fourier = soft_edged_kernel(radius=5, L=self.resolution, dim=2, in_fourier=True).to(device)

        with torch.no_grad():
            mask = mean_module.get_volume_mask()
            mean_volume = mean_module(None)
            idx = torch.arange(min(batch_size, len(self)), device=device)
            nufft_plan.setpts(pose_module(idx)[0])
            mask_threshold = get_mask_threshold(mask, nufft_plan) if mask is not None else 0
            pbar = tqdm(total=np.ceil(len(self) / batch_size), desc="Applying preprocessing on dataset images")
            for i in range(0, len(self), batch_size):
                idx = torch.arange(i, min(i + batch_size, len(self)))
                if pose_module.use_contrast:
                    # If pose module containts contrasts - apply them on filters
                    pts_rot, phase_shift, contrasts = pose_module(idx.to(device))
                    self.filters[idx] *= contrasts.reshape(-1, 1, 1).cpu()
                else:
                    pts_rot, phase_shift = pose_module(idx.to(device))

                images, _, filters, _ = self[idx]
                images = images.to(device)
                filters = filters.to(device)
                self.images[idx] = preprocess_image_batch(
                    images,
                    nufft_plan,
                    filters,
                    (pts_rot, phase_shift),
                    mean_volume,
                    mask,
                    mask_threshold,
                    softening_kernel_fourier,
                    fourier_domain=not self._in_spatial_domain,
                ).cpu()

                pbar.update(1)
            pbar.close()
        # After preprocessing images have no offsets or contrast variabillity.
        self.offsets[:] = 0
        self.contrasts = torch.ones((len(self)), dtype=self.offsets.dtype)

    def get_subset(self, idx: Iterable[int]) -> "CovarDataset":
        """Get a subset of the dataset.

        Args:
            idx: Indices of the subset

        Returns:
            Subset of the dataset
        """
        subset = self.copy()
        subset.images = subset.images[idx]
        subset.pts_rot = subset.pts_rot[idx]
        subset.filters = subset.filters[idx]
        subset.rot_vecs = subset.rot_vecs[idx]
        subset.offsets = subset.offsets[idx]

        return subset

    def half_split(self, permute: bool = True) -> Tuple["CovarDataset", "CovarDataset", torch.Tensor]:
        """Split the dataset into two halves.

        Args:
            permute: Whether to permute the dataset

        Returns:
            Two halves of the dataset and the permutation
        """
        data_size = len(self)
        if permute:
            permutation = torch.randperm(data_size)
        else:
            permutation = torch.arange(0, data_size)

        ds1 = self.get_subset(permutation[: data_size // 2])
        ds2 = self.get_subset(permutation[data_size // 2 :])

        return ds1, ds2, permutation

    def get_total_gain(self, batch_size: int = 1024, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns a 3D tensor represntaing the total gain of each frequency observed by the dataset
        = diag(sum(P_i^T P_i))

        Args:
            batch_size: Batch size for processing
            device: Device to use

        Returns:
            Total gain tensor
        """
        L = self.resolution
        upsample_factor = 1
        nufft_plan = NufftPlanDiscretized(
            (L,) * 3, upsample_factor=upsample_factor, mode="nearest", use_half_grid=False
        )
        device = get_torch_device() if device is None else device
        gain_tensor = torch.zeros((L * upsample_factor,) * 3, device=device, dtype=self.dtype)

        for i in range(0, len(self), batch_size):
            _, pts_rot, filters, _ = self[i : min(i + batch_size, len(self))]
            pts_rot = pts_rot.to(device)
            filters = filters.to(device)

            nufft_plan.setpts(pts_rot)

            gain_tensor += (
                im_backward(torch.complex(filters, torch.zeros_like(filters)), nufft_plan, filters, fourier_domain=True)
                .squeeze()
                .abs()
            )

        gain_tensor /= L

        return gain_tensor

    def get_total_covar_gain(
        self, batch_size: Optional[int] = None, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Returns a 2D tensor represnting the total gain of each frequency pair in the covariance
        least squares problem.

        Args:
            batch_size: Batch size for processing
            device: Device to use

        Returns:
            Total covariance gain tensor
        """

        L = self.resolution
        upsample_factor = 1
        nufft_plan = NufftPlanDiscretized(
            (L,) * 3, upsample_factor=upsample_factor, mode="nearest", use_half_grid=False
        )
        device = get_torch_device() if device is None else device
        gain_tensor = torch.zeros((L * upsample_factor,) * 3, device=device, dtype=self.dtype)

        s = average_fourier_shell(gain_tensor).shape[0]
        covar_shell_gain = torch.zeros((s, s), device=device, dtype=self.dtype)

        torch.cuda.empty_cache()

        if batch_size is None:
            total_mem = torch.cuda.get_device_properties(device).total_memory
            reserved_mem = torch.cuda.memory_reserved(device)
            available_memory = total_mem - reserved_mem
            available_batch_size = available_memory / (
                L**3 * self.dtype.itemsize * 2
            )  # self.dtype.itemsize*2 bytes per value for complex dtype

            batch_size = int(available_batch_size / 6)
            batch_size = min(batch_size, 256)
            logger.debug(f"Using batch size of {batch_size} to compute dataset covar gain")
            logger.debug(
                f"Memory reserved: {torch.cuda.memory_reserved(device)}, "
                f"Memory allocated: {torch.cuda.memory_allocated(device)}"
            )

        for i in range(0, len(self), batch_size):
            _, pts_rot, filters, _ = self[i : min(i + batch_size, len(self))]
            pts_rot = pts_rot.to(device)
            filters = filters.to(device)

            nufft_plan.setpts(pts_rot)

            if L % 2 == 0:
                filters[:, 0, :] = 0
                filters[:, :, 0] = 0
            gain_tensor = (
                nufft_plan.execute_adjoint_unaggregated(torch.complex(filters**2, torch.zeros_like(filters))).abs()
                / L**2
            )

            averaged_gain_tensor = average_fourier_shell(*gain_tensor)
            covar_shell_gain += averaged_gain_tensor.T @ averaged_gain_tensor

        torch.cuda.empty_cache()

        return covar_shell_gain

    def copy(self):
        return copy.deepcopy(self)

    def to_fourier_domain(self):
        if self._in_spatial_domain:
            self.images = centered_fft2(self.images)
            # TODO : transform points into grid_sample format here instead of in discretization function?
            self.noise_var *= (
                self.resolution**2
            )  # 2-d Fourier transform scales everything by a factor of L (and the variance scaled by L**2)
            self._in_spatial_domain = False

    def to_spatial_domain(self):
        if not self._in_spatial_domain:
            self.images = centered_ifft2(self.images).real
            self.noise_var /= self.resolution**2
            self._in_spatial_domain = True

    def estimate_signal_var(self, support_radius: Optional[float] = None, batch_size: int = 4096) -> None:
        """Estimate signal variance from the dataset.

        Args:
            support_radius: Radius for support region
            batch_size: Batch size for processing
        """
        # This computation should only be done once
        # the result is cached and returned if called again
        if self.signal_var is not None:
            return self.signal_var

        # Estimates the signal variance per pixel
        mask = torch.tensor(support_mask(self.resolution, support_radius))

        num_unmasked_pixels = 0
        signal_psd = torch.zeros((self.resolution, self.resolution))
        for i in range(0, len(self), batch_size):
            images_masked = self._get_images_for_signal_var(i, batch_size) * mask

            # We cannot use the size of the mask here, because the images may
            # already have been masked (with a tighter mask) prior to this
            # function call
            # TODO: this is not really a good way to get the number of unmasked pixels
            # ideally, we should cache the size of the projected mask
            images_threshold = (images_masked.abs() ** 2).mean(dim=(-1, -2)).sqrt() * 1e-6
            num_unmasked_pixels += (images_masked >= images_threshold.reshape(-1, 1, 1)).sum()

            signal_psd += torch.sum(torch.abs(centered_fft2(images_masked)) ** 2, axis=0)
        signal_psd /= (self.resolution**2) * num_unmasked_pixels
        signal_rpsd = average_fourier_shell(signal_psd)

        noise_psd = torch.ones((self.resolution, self.resolution)) * self.noise_var / (self.resolution**2)
        noise_rpsd = average_fourier_shell(noise_psd)

        radial_filters_gain = self.estimate_filters_gain()
        self.signal_rpsd = (signal_rpsd - noise_rpsd) / radial_filters_gain
        self.signal_rpsd[self.signal_rpsd < 0] = (
            0  # in low snr setting the estimatoin for high radial resolution might not be accurate enough
        )

        self.signal_var = sum_over_shell(self.signal_rpsd, self.resolution, 2).item()

        return self.signal_var

    def _get_images_for_signal_var(self, start_idx: int, batch_size: int) -> torch.Tensor:
        """Helper method to get images for signal variance estimation.

        Subclasses can override this to provide different image access patterns.
        """
        end_idx = min(start_idx + batch_size, len(self))
        return self.images[start_idx:end_idx]

    def estimate_filters_gain(self, batch_size: int = 4096) -> None:
        """Estimate gain factors for CTF filters.

        Args:
            batch_size: Batch size for processing
        """
        # This computation should only be done once
        # the result is cached and returned if called again
        if self.radial_filters_gain is not None:
            return self.radial_filters_gain

        average_filters_gain_spectrum = torch.zeros((self.resolution, self.resolution))
        for i in range(0, len(self), batch_size):
            filters = self._get_filters_for_filters_gain(i, batch_size)
            average_filters_gain_spectrum += torch.sum(filters**2, axis=0)
        average_filters_gain_spectrum /= len(self)

        radial_filters_gain = average_fourier_shell(average_filters_gain_spectrum)

        self.radial_filters_gain = radial_filters_gain

        return self.radial_filters_gain

    def _get_filters_for_filters_gain(self, start_idx: int, batch_size: int) -> torch.Tensor:
        """Helper method to get filters for filter gain estimation.

        Subclasses can override this to provide different CTF access patterns.
        """
        end_idx = min(start_idx + batch_size, len(self))
        return self.filters[start_idx:end_idx]

    def update_pose(self, pose_module: PoseModule, batch_size: int = 1024) -> None:
        """Updates dataset's particle pose information from a given PoseModule.

        Args:
            pose_module (PoseModule): PoseModule instance to update from
        """
        with torch.no_grad():
            for i in range(0, len(self), batch_size):
                idx = torch.arange(i, min(i + batch_size, len(self)), device=pose_module.device)
                self.pts_rot[idx.cpu()] = pose_module(idx)[0].detach().cpu()
                self.offsets[idx.cpu()] = pose_module.get_offsets()[idx.cpu()].detach().cpu()
                if pose_module.use_contrast:
                    self.contrasts[idx.cpu()] = pose_module.get_contrasts()[idx.cpu()].detach().cpu().squeeze()


class LazyCovarDataset(CovarDataset):
    def __init__(self, src, noise_var, mean_volume=None, mask=None, invert_data=False, apply_preprocessing=True):
        if not isinstance(src, ImageSource):
            raise ValueError(f"input src is of type {type(src)}. LazyCovarDataset only supports ImageSource")
        self.src = src
        self.resolution = self.src.resolution
        self.noise_var = noise_var
        self.data_inverted = invert_data
        self._in_spatial_domain = True
        self.apply_preprocessing = apply_preprocessing

        self.mask = torch.tensor(mask.asnumpy()) if mask is not None else None
        self.mean_volume = mean_volume

        # Decalare additional attributes that will be initialized in post_init_setup
        self._mean_volume = None
        self._mask = None
        self._pose_module = None
        self._nufft_plan = None
        self._softening_kernel_fourier = None
        self._mask_threshold = None
        self.contrasts = torch.ones((len(self)), dtype=self.offsets.dtype)

        self.radial_filters_gain = None
        self.signal_rpsd = None
        self.signal_var = None

    @property
    def dtype(self):
        return self.src.dtype

    def post_init_setup(self, fourier_domain):
        """Performs additional setup after constructor.

        It inits a nufft plan that is used internally to compute projections of the mean volume and
        mask. This must happen after class construction since when we use DDP we pass this object
        which cannot have tensors already on the GPU.
        """
        # TODO: should better handle case where apply_preprocessing=False, this will use uncessery GPU mem
        rot_vecs = torch.tensor(
            Rotation(self.src.rotations).as_rotvec(), dtype=self.dtype
        )  # TODO: use a torch implementation?
        mean_module, pose_module, nufft_plan = self.construct_mean_pose_modules(
            self.mean_volume, self.mask, rot_vecs, self.src.offsets, fourier_domain=fourier_domain
        )
        self._set_internal_preprocessing_modules(mean_module, pose_module, nufft_plan)

    def preprocess_from_modules(self, mean_module, pose_module, nufft_plan=None, batch_size=1024):
        """Overrides superclass method, since this is a lazy dataset implementation, this does not
        actually perform any preprocessing but it update the internal objects required preprocessing
        on demand."""

        pose_module = copy.deepcopy(pose_module)

        if mean_module is not None:
            mean_module = copy.deepcopy(mean_module)

            if mean_module._in_spatial_domain != self._in_spatial_domain:
                domain_name = lambda val: "Spatial" if val else "Fourier"
                logger.warning(
                    f"Mean module is in {domain_name(mean_module._in_spatial_domain)} domain "
                    f"while dataset is in {domain_name(self._in_spatial_domain)}. "
                    "Changing domain of the mean module to fit dataset"
                )

                mean_module._in_spatial_domain = self._in_spatial_domain

            device = get_torch_device()
            if nufft_plan is None:
                nufft_plan = (
                    NufftPlan((self.resolution,) * 3, batch_size=1, dtype=mean_module.dtype, device=device)
                    if mean_module._in_spatial_domain
                    else NufftPlanDiscretized(
                        (self.resolution,) * 3, upsample_factor=mean_module.upsampling_factor, mode="bilinear"
                    )
                )

        if not self.apply_preprocessing:
            logger.debug("Apply pre-preprocessing is False, when preprocess_from_modoules was called. Changing to True")
            self.apply_preprocessing = True

        self._set_internal_preprocessing_modules(mean_module, pose_module, nufft_plan)

    def _set_internal_preprocessing_modules(self, mean_module, pose_module, nufft_plan):
        set_module_grad(pose_module, False)
        device = get_torch_device()
        self.src = self.src.to(device)

        self._pose_module = pose_module.to(device)
        self._nufft_plan = nufft_plan

        if mean_module is not None:
            set_module_grad(mean_module, False)
            mean_module = mean_module.to(device)

            # _mean_volume and _mask are different than the original mean_volume and mask
            # in that they are in fourier domain(if fourier_domain is True)
            self._mean_volume = mean_module()
            self._mask = mean_module.get_volume_mask()

            # Compute mask related variables
            with torch.no_grad():
                idx = torch.arange(min(1024, len(self)), device=device)
                nufft_plan.setpts(pose_module(idx)[0])
                self._mask_threshold = get_mask_threshold(self._mask, nufft_plan) if self._mask is not None else 0
                # Mask softening kernel should be in fourier space regardless of the value of fourier_domain
                self._softening_kernel_fourier = soft_edged_kernel(
                    radius=5, L=self.resolution, dim=2, in_fourier=True
                ).to(device)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self._get_images(idx) + (idx,)

    def _get_images(
        self, idx: Iterable, filters: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns images by idx after pre-processing (if needed) which includes substracting
        projected mean and masking.

        Args:
            idx (Iterable): image index to be returned
            filters (Optional[torch.Tensor]): CTF filters to be used when computing the mean projection.
                If None are given CTFs are computed by the corresponding image idx.

        Returns:
            Tuple[torch.Tensor,torch.Tensor,torch.Tensor]: Tuple containing images,rotated grid of fourier components
            and filters.
        """
        device = self.src.device
        images = self.src.images(idx, fourier=not self._in_spatial_domain)

        image_sign = -1 if self.data_inverted else 1
        images *= image_sign

        # Compute CTF if not provided
        if filters is None:
            filters = self.src.get_ctf(idx)

        if not isinstance(idx, torch.Tensor):
            if isinstance(idx, slice):
                idx = torch.arange(len(self.src))[idx]
            else:
                idx = torch.tensor(idx)

        idx = idx.to(device)

        # Compute pts
        if self._pose_module.use_contrast:
            pts_rot, phase_shift, contrasts = self._pose_module(idx)
            filters = filters * contrasts.reshape(
                -1, 1, 1
            )  # TODO: need to handle case where filters are provided, should contrast be applied?
        else:
            pts_rot, phase_shift = self._pose_module(idx)

        if not self.apply_preprocessing:
            return images, pts_rot, filters

        images = images.to(device)
        filters = filters.to(device)

        with torch.no_grad():
            images = preprocess_image_batch(
                images,
                self._nufft_plan,
                filters,
                (pts_rot, phase_shift),
                self._mean_volume,
                self._mask,
                self._mask_threshold,
                self._softening_kernel_fourier,
                fourier_domain=not self._in_spatial_domain,
            )

        return images, pts_rot, filters

    @property
    def filters(self):
        return self.src.get_ctf(torch.arange(0, len(self)))

    @property
    def rot_vecs(self):
        return torch.tensor(Rotation(self.src.rotations.numpy()).as_rotvec(), dtype=self.src.rotations.dtype)

    @property
    def offsets(self):
        return self.src.offsets

    def _get_images_for_signal_var(self, start_idx, batch_size):
        """Override to use lazy loading for signal variance estimation."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.src.images(torch.arange(start_idx, end_idx)).cpu()

    def _get_filters_for_filters_gain(self, start_idx, batch_size):
        """Override to use lazy loading for signal variance estimation."""
        end_idx = min(start_idx + batch_size, len(self))
        return self.src.get_ctf(torch.arange(start_idx, end_idx)).cpu()

    def get_subset(self, idx):
        subset = self.copy()
        subset.src = self.src.get_subset(idx)

        # Create a new pose module for the subset
        if self._pose_module is not None:
            rot_vecs = torch.tensor(Rotation(subset.src.rotations).as_rotvec(), dtype=self.dtype)
            offsets = subset.src.offsets
            subset._pose_module = PoseModule(rot_vecs, offsets, self.resolution)

        return subset

    def half_split(self, permute: bool = False) -> Tuple["LazyCovarDataset", "LazyCovarDataset", torch.Tensor]:
        assert not permute, "Cannot permute dataset in lazy mode"
        return super().half_split(False)

    def to_fourier_domain(self):
        if self._in_spatial_domain:
            self.post_init_setup(fourier_domain=True)
            self._in_spatial_domain = False

    def to_spatial_domain(self):
        if not self._in_spatial_domain:
            self.post_init_setup(fourier_domain=False)
            self._in_spatial_domain = True

    def update_pose(self, pose_module: PoseModule, batch_size: int = None):
        """Updates dataset's particle pose information from a given PoseModule.

        Args:
            pose_module (PoseModule): PoseModule instance to update from
        """
        rot_vecs = pose_module.get_rotvecs().detach()
        offsets = pose_module.get_offsets().detach().cpu()

        self.src.rotations = rotvec_to_rotmat(rot_vecs).cpu()
        self.src.offsets = offsets

        # Update pose on exisiting internal pose module
        if self._pose_module is not None:
            self._pose_module.set_rotvecs(rot_vecs)
            self._pose_module.set_offsets(offsets)

        if pose_module.use_contrast:
            # We do not want to update internal pose module contrast
            # since it will modify the returned filters on _get_images
            self.contrasts = pose_module.get_contrasts().detach().cpu()


def get_ordered_half_split(n: int) -> torch.Tensor:
    """Split n indices into two random halves and keep each half ordered. This is usefull for lazy
    dataset where reading Truly shuffled data can be very slow.

    Args:
        n (int): Total number of elements to split

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two sorted tensors containing the indices for each half
    """
    idx = torch.randperm(n)
    half1 = idx[: n // 2]
    half2 = idx[n // 2 :]
    # Order the two halves (smallest index first in each half)
    half1, _ = torch.sort(half1)
    half2, _ = torch.sort(half2)
    return half1, half2


def is_dataset_lazy(dataset: CovarDataset) -> bool:
    """Check if a CovarDataset instance is an instance of LazyCovarDataset.

    Args:
        dataset (CovarDataset): The dataset to check.

    Returns:
        bool: True if the dataset is a LazyCovarDataset, False otherwise.
    """
    return isinstance(dataset, LazyCovarDataset)


class BatchIndexSampler(torch.utils.data.Sampler):
    """Custom sampler for batching indices with optional shuffling.

    Attributes:
        data_size: Total number of data points
        batch_size: Size of each batch
        shuffle: Whether to shuffle indices
        idx: List of indices to sample from
    """

    def __init__(self, data_size: int, batch_size: int, shuffle: bool = True, idx: Optional[List[int]] = None) -> None:
        self.data_size = data_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        if idx is None:
            idx = list(range(self.data_size))
        else:
            self.data_size = len(idx)
        if self.shuffle:
            random.shuffle(idx)
        self.idx = torch.tensor(idx)

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate over batches of indices.

        Yields:
            Batches of indices as tensors
        """
        for i in range(0, self.data_size, self.batch_size):
            yield self.idx[i : i + self.batch_size]

    def __len__(self) -> int:
        """Return the number of batches.

        Returns:
            Number of batches
        """
        return (self.data_size + self.batch_size - 1) // self.batch_size


def identity_collate(batch: List[Any]) -> List[Any]:
    """Identity collate function that returns batch as-is.

    Args:
        batch: List of items to collate

    Returns:
        Same list of items
    """
    return batch


def create_dataloader(
    dataset: torch.utils.data.Dataset, batch_size: int, idx: Optional[List[int]] = None, **dataloader_kwargs: Any
) -> torch.utils.data.DataLoader:
    """Create a DataLoader with custom batch sampling.

    Args:
        dataset: Dataset to create loader for
        batch_size: Size of each batch
        idx: Optional list of indices to sample from
        **dataloader_kwargs: Additional arguments for DataLoader

    Returns:
        Configured DataLoader instance
    """
    sampler = dataloader_kwargs.pop("sampler", None)

    if sampler is None:
        sampler = BatchIndexSampler(len(dataset), batch_size, shuffle=False, idx=idx)
    else:
        sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    # Cannot use num workers > 1 with lazy dataset since it requires GPU usage
    # TODO: find a better solution that enables the use of more workers
    num_workers = dataloader_kwargs.pop("num_workers", 0)
    if isinstance(dataset, LazyCovarDataset):
        logger.debug(
            f"Warning: cannot use {num_workers} > 1 num_workers with Lazy dataset. "
            "Setting num_workers to 0 and prefetch_factor to None"
        )
        dataloader_kwargs["prefetch_factor"] = None
        dataloader_kwargs["persistent_workers"] = False
        dataloader_kwargs["pin_memory"] = False
        dataloader_kwargs["pin_memory_device"] = ""
        num_workers = 0

    batch_size = None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        # Can't use lambda function here since it will be pickled
        # and sent to other processes when using DDP
        collate_fn=identity_collate,
        **dataloader_kwargs,
    )


def get_dataloader_batch_size(dataloader):
    batch_size = dataloader.batch_size
    if batch_size is None:
        batch_size = dataloader.sampler.batch_size
    return batch_size


@dataclass
class GTData:
    """Class to hold the ground truth data to compute metrics against ground truth."""

    eigenvecs: torch.Tensor = None
    mean: torch.Tensor = None
    rotations: torch.Tensor = None
    offsets: torch.Tensor = None
    contrasts: torch.Tensor = None

    def __post_init__(self):
        if self.eigenvecs is not None:
            self.eigenvecs = torch.tensor(self.eigenvecs)
        if self.mean is not None:
            self.mean = torch.tensor(self.mean)
        if self.rotations is not None:
            self.rotations = torch.tensor(self.rotations)
        if self.offsets is not None:
            self.offsets = torch.tensor(self.offsets)
        if self.contrasts is not None:
            self.contrasts = torch.tensor(self.contrasts)

    def half_split(self, permutation=None):
        rotations_present = self.rotations is not None
        offsets_present = self.offsets is not None
        contrasts_present = self.contrasts is not None
        if not (rotations_present or offsets_present):
            return self, self

        n = self.rotations.shape[0] if rotations_present else self.offsets.shape[0]
        if permutation is None:
            permutation = torch.arange(n)
        perm = permutation[: n // 2], permutation[n // 2 :]

        rotations1 = self.rotations[perm[0]] if rotations_present else None
        offsets1 = self.offsets[perm[0]] if offsets_present else None
        contrasts1 = self.contrasts[perm[0]] if contrasts_present else None
        rotations2 = self.rotations[perm[1]] if rotations_present else None
        offsets2 = self.offsets[perm[1]] if offsets_present else None
        contrasts2 = self.contrasts[perm[1]] if contrasts_present else None

        gt1 = GTData(
            eigenvecs=self.eigenvecs,
            mean=self.mean,
            rotations=rotations1,
            offsets=offsets1,
            contrasts=contrasts1,
        )
        gt2 = GTData(
            eigenvecs=self.eigenvecs,
            mean=self.mean,
            rotations=rotations2,
            offsets=offsets2,
            contrasts=contrasts2,
        )
        return gt1, gt2
