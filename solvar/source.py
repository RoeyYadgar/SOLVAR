import os
import pickle
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import mrcfile
import numpy as np
import torch
from aspire.image import Image
from aspire.operators import ArrayFilter, MultiplicativeFilter, ScalarFilter
from aspire.utils import Rotation
from aspire.volume import Volume, rotated_grids
from cryodrgn.ctf import compute_ctf, load_ctf_for_training
from cryodrgn.source import ImageSource as CryoDRGNImageSource

from solvar.nufft_plan import NufftPlan, NufftSpec
from solvar.poses import get_phase_shift_grid, pose_ASPIRE2cryoDRGN, pose_cryoDRGN2APIRE
from solvar.projection_funcs import centered_fft2, centered_ifft2, make_nufft_plan, vol_forward
from solvar.utils import get_torch_device


class ImageSource:
    """Image source for cryo-EM particle images with CTF and pose information.

    Provides access to particle images along with their corresponding CTF parameters,
    rotations, and translations. Supports preprocessing including whitening and normalization.
    Uses cryoDRGN image source under the hood to read particle images and associated data.

    Attributes:
        particles_path: Path to particle images file
        ctf_path: Path to CTF parameters file
        poses_path: Path to poses file
        device: Device for computations
        image_source: CryoDRGN image source
        dtype: Data type for tensors
        ctf_params: CTF parameters for each particle
        freq_lattice: Frequency lattice for CTF computation
        indices: Indices of particles to use
        rotations: Rotation matrices for each particle
        offsets: Translation offsets for each particle
        apply_preprocessing: Whether to apply preprocessing
        whitening_filter: Whitening filter for noise reduction
        offset_normalization: Per-image offset normalization
        scale_normalization: Per-image scale normalization
    """

    def __init__(
        self,
        particles_path: str,
        ctf_path: Optional[str] = None,
        poses_path: Optional[str] = None,
        indices: Optional[torch.Tensor] = None,
        apply_preprocessing: bool = True,
    ) -> None:
        """Initialize ImageSource.

        Args:
            particles_path: Path to particle images file
            ctf_path: Path to CTF parameters file (optional)
            poses_path: Path to poses file (optional)
            indices: Indices of particles to use (optional)
            apply_preprocessing: Whether to apply preprocessing (default: True)
        """
        self.particles_path = particles_path
        self.device = torch.device("cpu")
        self.image_source = CryoDRGNImageSource.from_file(self.particles_path, indices=indices)
        if self.image_source.dtype == "float32" or self.image_source.dtype == np.float32:
            self.dtype = torch.float32
        elif self.image_source.dtype == "float64" or self.image_source.dtype == np.float64:
            self.dtype = torch.float64
        else:
            raise ValueError(f"Unsupported dtype: {self.image_source.dtype}. Only float32 and float64 are supported.")

        # If ctf or poses were not provided check if they exist in the same dir as the particles file
        particles_dir = os.path.split(self.particles_path)[0]
        if ctf_path is None:
            ctf_path = os.path.join(particles_dir, "ctf.pkl")
            assert os.path.isfile(
                ctf_path
            ), f"ctf file was not provided, tried {ctf_path} as a default but file does not exist"
        if poses_path is None:
            poses_path = os.path.join(particles_dir, "poses.pkl")
            assert os.path.isfile(
                poses_path
            ), f"poses file was not provided, tried {poses_path} as a default but file does not exist"
        self.ctf_path = ctf_path
        self.poses_path = poses_path

        self.ctf_params = torch.tensor(load_ctf_for_training(self.resolution, ctf_path))
        self.freq_lattice = (
            (torch.stack(get_phase_shift_grid(self.resolution), dim=0) / torch.pi / 2)
            .permute(2, 1, 0)
            .reshape(self.resolution**2, 2)
        )

        if indices is None:
            indices = torch.arange(self.image_source.n)
        self.indices = indices
        self.ctf_params = self.ctf_params[indices]

        with open(poses_path, "rb") as f:
            poses = pickle.load(f)
        rots, offsets = pose_cryoDRGN2APIRE(poses, self.resolution)
        self.rotations = torch.tensor(rots.astype(self.image_source.dtype))[indices]
        self.offsets = torch.tensor(offsets.astype(self.image_source.dtype))[indices]
        self.apply_preprocessing = apply_preprocessing

        self.whitening_filter = None
        self.offset_normalization = torch.zeros(self.image_source.n)
        self.scale_normalization = torch.ones(self.image_source.n)
        if self.apply_preprocessing:
            self._preprocess_images()

    @property
    def resolution(self) -> int:
        """Get image resolution.

        Returns:
            Image resolution
        """
        return self.image_source.D

    def __len__(self) -> int:
        """Get number of particles.

        Returns:
            Number of particles
        """
        return self.image_source.n

    def to(self, device: Union[torch.device, str, None]) -> "ImageSource":
        """Move the ImageSource to the specified device (CPU/GPU).

        Args:
            device: Target device

        Returns:
            Self for method chaining
        """
        if device is None:
            device = torch.device("cpu")

        self.device = device

        # Move tensors to device
        self.ctf_params = self.ctf_params.to(device)
        self.freq_lattice = self.freq_lattice.to(device)
        self.offset_normalization = self.offset_normalization.to(device)
        self.scale_normalization = self.scale_normalization.to(device)

        # Move whitening filter if it exists
        if self.whitening_filter is not None:
            self.whitening_filter = self.whitening_filter.to(device)

        return self

    def get_ctf(self, index: Union[int, torch.Tensor]) -> torch.Tensor:
        """Get CTF for given particle indices.

        Args:
            index: Particle index or indices

        Returns:
            CTF values
        """
        ctf_params = self.ctf_params[index]
        freq_lattice = self.freq_lattice / ctf_params[:, 0].view(-1, 1, 1)
        ctf = compute_ctf(freq_lattice, *torch.split(ctf_params[:, 1:], 1, 1)).reshape(
            -1, self.resolution, self.resolution
        )

        return ctf if not self.apply_preprocessing else ctf * self.whitening_filter

    def images(self, index: Union[int, torch.Tensor], fourier: bool = False) -> torch.Tensor:
        """Get particle images for given indices.

        Args:
            index: Particle index or indices
            fourier: Whether to return in Fourier domain (default: False)

        Returns:
            Particle images
        """
        images = self.image_source.images(index)
        images = images.to(self.device)
        if not self.apply_preprocessing and not fourier:
            return images

        images = centered_fft2(images)

        if self.apply_preprocessing:
            images *= self.whitening_filter
            images[:, self.resolution // 2, self.resolution // 2] -= (
                self.offset_normalization[index] * self.resolution**2
            )
            images /= self.scale_normalization[index].reshape(-1, 1, 1)

        if not fourier:
            images = centered_ifft2(images).real

        return images

    def __getitem__(
        self, index: Union[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get particle data for given index.

        Args:
            index: Particle index

        Returns:
            Tuple of (images, ctf, rotations, offsets)
        """
        return self.images(index), self.get_ctf(index), self.rotations[index], self.offsets[index]

    def _preprocess_images(self, batch_size: int = 1024) -> None:
        """Preprocess images with whitening and normalization.

        Whitens images by estimating the noise PSD and apply it as a filter on all images.
        Additionally each image is normalized individually to have N(0,1) background noise.
        Implementation is based on ASPIRE:
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/noise/noise.py#L333
            https://github.com/ComputationalCryoEM/ASPIRE-Python/blob/main/src/aspire/image/image.py#L27

        Args:
            batch_size: Batch size for processing (default: 1024)
        """
        mask = (torch.norm(self.freq_lattice, dim=1) >= 0.5).reshape(self.resolution, self.resolution)
        n = len(self)
        mean_est = 0
        noise_psd_est = torch.zeros((self.resolution,) * 2)
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            # Use original unaltered images (not self.images)
            images = self.image_source.images(idx) * mask

            mean_est += torch.sum(images)
            noise_psd_est += torch.sum(torch.abs(centered_fft2(images)) ** 2, dim=0)

        mean_est /= torch.sum(mask) * n
        noise_psd_est /= torch.sum(mask) * n

        noise_psd_est[self.resolution // 2, self.resolution // 2] -= mean_est**2

        self.whitening_filter = (1 / torch.sqrt(noise_psd_est)).unsqueeze(0)

        # Per-image normalization
        # After setting up whitening filter, we can access self.images to get the whitened images
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            images = self.images(idx)
            mean = torch.mean(images[:, mask], dim=1)
            std = torch.std(images[:, mask], dim=1)
            self.offset_normalization[idx] = mean
            self.scale_normalization[idx] = std

    def estimate_noise_var(self, batch_size: int = 1024) -> float:
        """Estimate noise variance from images.

        Args:
            batch_size: Batch size for processing (default: 1024)

        Returns:
            Estimated noise variance
        """
        mask = (torch.norm(self.freq_lattice, dim=1) >= 0.5).reshape(self.resolution, self.resolution)
        n = len(self)
        first_moment = 0
        second_moment = 0
        for i in range(0, n, batch_size):
            idx = torch.arange(i, min(i + batch_size, n))
            images = self.images(idx)
            images_masked = images * mask

            first_moment += torch.sum(images_masked)
            second_moment += torch.sum(torch.abs(images_masked) ** 2)

        first_moment /= torch.sum(mask) * n
        second_moment /= torch.sum(mask) * n
        return second_moment - first_moment**2

    def get_subset(self, idx: Union[int, torch.Tensor]) -> "ImageSource":
        """Get subset of particles.

        Args:
            idx: Indices of particles to include

        Returns:
            New ImageSource with subset of particles
        """
        subset = deepcopy(self)
        subset.indices = subset.indices[idx]
        subset.image_source = CryoDRGNImageSource.from_file(subset.particles_path, indices=subset.indices)
        subset.ctf_params = subset.ctf_params[idx]
        subset.rotations = subset.rotations[idx]
        subset.offsets = subset.offsets[idx]

        subset.scale_normalization = subset.scale_normalization[idx]
        subset.offset_normalization = subset.offset_normalization[idx]

        return subset

    def get_paths(self) -> Tuple[str, str, str]:
        """Get file paths.

        Returns:
            Tuple of (particles_path, ctf_path, poses_path)
        """
        return self.particles_path, self.ctf_path, self.poses_path


class SimulatedSource:
    """Simulated source for generating synthetic cryo-EM heterogeneity dataset. Uses the same
    interface as ASPIRE ImageSource needed to be consumed by CovarDataset.

    Attributes:
        n: Number of particles to generate
        L: Image resolution
        num_vols: Number of volumes
        vols: Volume data
        whiten: Whether to whiten the images
        _unique_filters: List of unique CTF filters
        rotations_std: Standard deviation for rotation noise
        offsets_std: Standard deviation for offset noise
        _clean_images: Clean images without noise
        _noise_var: Noise variance
        _image_noise: Noise tensor
        offsets: Translation offsets
        amplitudes: Amplitude scaling factors
        states: Volume state assignments
        filter_indices: Filter index assignments
        rotations: Rotation matrices
        _rotations: Original rotation matrices without noise
        _offsets: Original offsets without noise
    """

    def __init__(
        self,
        n: int,
        vols: Volume,
        noise_var: float,
        whiten: bool = True,
        unique_filters: Optional[List] = None,
        rotations_std: float = 0,
        offsets_std: float = 0,
        nufft_spec: Optional[NufftSpec] = None,
    ) -> None:
        self.n = n
        self.L = vols.shape[-1]
        self.num_vols = vols.shape[0]
        self.vols = vols
        self.whiten = whiten
        if unique_filters is None:
            unique_filters = [ArrayFilter(np.ones((self.L, self.L)))]
        self._unique_filters = unique_filters
        self.rotations_std = rotations_std
        self.offsets_std = offsets_std
        self.np_dtype = vols.asnumpy().dtype
        self.dtype = torch.tensor(vols.asnumpy()).dtype

        # Nufft spec is used to determine how to project input volumes into the dataset images
        # default is standard nufft
        if nufft_spec is None:
            nufft_spec = NufftSpec(NufftPlan, (self.L,) * 3, batch_size=1, dtype=self.dtype, device=get_torch_device())
        self.nufft_spec = nufft_spec

        self._clean_images = self._gen_clean_images()
        self.noise_var = noise_var

    @property
    def noise_var(self) -> float:
        """Get effective noise variance.

        Returns:
            Noise variance (1.0 if whitened, actual value otherwise)
        """
        return self._noise_var if (not self.whiten) else 1

    @noise_var.setter
    def noise_var(self, noise_var: float) -> None:
        """Set noise variance and generate corresponding noise.

        Args:
            noise_var: Noise variance value
        """
        self._noise_var = noise_var
        self._image_noise = torch.randn(
            (self.n, self.L, self.L), dtype=self._clean_images.dtype, device=self._clean_images.device
        ) * (self._noise_var**0.5)

    @property
    def images(self) -> Image:
        """Get noisy images with optional whitening.

        Returns:
            ASPIRE Image object containing noisy particle images
        """
        images = self._clean_images + self._image_noise
        if self.whiten:
            images /= (self._noise_var) ** 0.5

        return Image(images.numpy())

    @property
    def unique_filters(self) -> List:
        """Get unique filters with whitening applied.

        Returns:
            List of filters with whitening filter applied
        """
        if self.whiten:
            whiten_filter = ScalarFilter(dim=2, value=self._noise_var ** (-0.5))
            return [MultiplicativeFilter(filt, whiten_filter) for filt in self._unique_filters]

        return self._unique_filters

    def noisify_rotations(self, rots: np.ndarray, noise_std: float) -> np.ndarray:
        """Add noise to rotation matrices.

        Args:
            rots: Input rotation matrices
            noise_std: Standard deviation of rotation noise

        Returns:
            Noisy rotation matrices
        """
        noisy_rots = Rotation.from_matrix(rots).as_rotvec()
        noisy_rots += noise_std * np.random.randn(*noisy_rots.shape)
        return Rotation.from_rotvec(noisy_rots).matrices.astype(rots.dtype)

    def _gen_clean_images(self, batch_size: int = 1024) -> torch.Tensor:
        """Generate clean images by projecting volumes.

        Args:
            batch_size: Batch size for processing

        Returns:
            Clean images tensor
        """
        clean_images = torch.zeros((self.n, self.L, self.L), dtype=self.dtype)
        self._offsets = torch.zeros((self.n, 2))  # TODO: create non-zero gt offsets
        self.offsets = self._offsets + self.L * self.offsets_std * np.random.randn(self.n, 2)
        self.amplitudes = np.ones((self.n))
        self.states = torch.tensor(np.random.choice(self.num_vols, self.n))
        self.filter_indices = np.random.choice(len(self._unique_filters), self.n)
        self._rotations = Rotation.generate_random_rotations(self.n, dtype=self.np_dtype).matrices
        self.rotations = self.noisify_rotations(self._rotations, self.rotations_std)

        unique_filters = torch.tensor(
            np.array([self._unique_filters[i].evaluate_grid(self.L) for i in range(len(self._unique_filters))]),
            dtype=self.dtype,
        )
        pts_rot = torch.tensor(rotated_grids(self.L, self._rotations).copy()).reshape((3, self.n, self.L**2))
        pts_rot = pts_rot.transpose(0, 1)
        pts_rot = torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi

        device = get_torch_device()
        volumes = torch.tensor(self.vols.asnumpy(), device=device)
        nufft_plan, volumes = make_nufft_plan(self.nufft_spec, volumes)

        for i in range(self.num_vols):
            idx = (self.states == i).nonzero().reshape(-1)
            for j in range(0, len(idx), batch_size):
                batch_ind = idx[j : j + batch_size]
                ptsrot = pts_rot[batch_ind].to(device)
                filter_indices = self.filter_indices[batch_ind]
                filters = unique_filters[filter_indices].to(device)

                nufft_plan.setpts(ptsrot)
                projected_volume = vol_forward(volumes[i].unsqueeze(0), nufft_plan, filters).squeeze(1)

                clean_images[batch_ind] = projected_volume.cpu()

        return clean_images

    def _ctf_cryodrgn_format(self):
        ctf = np.zeros((len(self._unique_filters), 9))
        for i, ctf_filter in enumerate(self._unique_filters):
            ctf[i, 0] = self.L
            ctf[i, 1] = ctf_filter.pixel_size
            ctf[i, 2] = ctf_filter.defocus_u
            ctf[i, 3] = ctf_filter.defocus_v
            ctf[i, 4] = ctf_filter.defocus_ang / np.pi * 180
            ctf[i, 5] = ctf_filter.voltage
            ctf[i, 6] = ctf_filter.Cs
            ctf[i, 7] = ctf_filter.alpha
            ctf[i, 8] = 0  # phase shift

        full_ctf = np.zeros((self.n, 9))
        for i in range(ctf.shape[0]):
            full_ctf[self.filter_indices == i] = ctf[i]

        return full_ctf

    def save(
        self,
        output_dir: str,
        file_prefix: Optional[str] = None,
        save_image_stack: bool = True,
        gt_pose: bool = False,
        whiten: bool = False,
    ) -> None:
        """Save simulated data to files.

        Args:
            output_dir: Output directory for saved files
            file_prefix: Optional prefix for filenames
            save_image_stack: Whether to save image stack
            gt_pose: Whether to use ground truth poses
            whiten: Whether to apply whitening
        """

        def add_prefix(filename: str) -> str:
            return f"{file_prefix}_{filename}" if file_prefix is not None else filename

        mrcs_output = os.path.join(output_dir, add_prefix("particles.mrcs"))
        poses_output = os.path.join(output_dir, add_prefix("poses.pkl"))
        ctf_output = os.path.join(output_dir, add_prefix("ctf.pkl"))

        if save_image_stack:
            whiten_val = self.whiten
            self.whiten = whiten
            with mrcfile.new(mrcs_output, overwrite=True) as mrc:
                mrc.set_data(self.images.asnumpy().astype(np.float32))
                # mrc.voxel_size = self.vols.pixel_size
                # mrc.set_spacegroup(1)
                # mrc.data = np.transpose(mrc.data,(0,2,1))
                # mrc.update_header()
            self.whiten = whiten_val

        if gt_pose:
            rots = self._rotations
            offsets = self._offsets
        else:
            rots = self.rotations
            offsets = self.offsets
        poses = pose_ASPIRE2cryoDRGN(rots, offsets, self.L)
        with open(poses_output, "wb") as f:
            pickle.dump(poses, f)

        with open(ctf_output, "wb") as f:
            pickle.dump(self._ctf_cryodrgn_format(), f)
