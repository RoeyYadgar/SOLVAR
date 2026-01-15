from typing import Optional, Tuple, Union

import numpy as np
import torch
from aspire.utils import grid_2d

from solvar.newton_opt import BlockNewtonOptimizer
from solvar.projection_funcs import centered_fft2, centered_ifft2, crop_image


def pose_cryoDRGN2APIRE(poses: Tuple[np.ndarray, np.ndarray], L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert poses from cryoDRGN format to ASPIRE format.

    Args:
        poses: Tuple of (rotations, offsets) in cryoDRGN format
        L: Image resolution

    Returns:
        Tuple of (rotations, offsets) in ASPIRE format
    """
    rots = np.transpose(poses[0], axes=(0, 2, 1))
    offsets = poses[1] * L

    return rots, offsets


def pose_ASPIRE2cryoDRGN(rots: np.ndarray, offsets: np.ndarray, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert poses from ASPIRE format to cryoDRGN format.

    Args:
        rots: Rotation matrices in ASPIRE format
        offsets: Translation offsets in ASPIRE format
        L: Image resolution

    Returns:
        Tuple of (rotations, offsets) in cryoDRGN format
    """
    rots = np.transpose(rots, axes=(0, 2, 1))
    offsets = offsets / L

    return (rots, offsets)


def pad_poses_by_ind(poses: Tuple[np.ndarray, np.ndarray], ind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Pad pose arrays to a full set of particle indices.

    Given a tuple of pose arrays (rotations and offsets) and an index array,
    this function constructs zero-padded pose arrays sized for the full range
    up to the largest index in `ind`. The given poses are inserted at the
    corresponding indices.

    Args:
        poses: Tuple of (rotations, offsets), both already corresponding to `ind`.
        ind: 1D integer array of indices specifying positions where poses should be inserted.

    Returns:
        full_poses: Tuple of (rotations, offsets), both arrays padded to shape
                    (max(ind), ...) with the given values at `ind` and zeros elsewhere.
    """
    n = np.max(ind) + 1
    full_poses = (np.zeros((n, 3, 3), dtype=poses[0].dtype), np.zeros((n, 2), dtype=poses[1].dtype))

    full_poses[0][ind] = poses[0]
    full_poses[1][ind] = poses[1]

    return full_poses


def rotvec_to_rotmat(rotvecs: torch.Tensor) -> torch.Tensor:
    """Convert rotation vectors to rotation matrices using Rodrigues' formula.

    Args:
        rotvecs: Rotation vectors of shape (N, 3)

    Returns:
        Rotation matrices of shape (N, 3, 3)
    """
    theta = torch.norm(rotvecs, dim=-1, keepdim=True)  # (N, 1)
    k = rotvecs / (theta + 1e-6)  # Normalize, avoiding division by zero

    K = torch.zeros((rotvecs.shape[0], 3, 3), device=rotvecs.device, dtype=rotvecs.dtype)  # (N, 3, 3)
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]

    eye = torch.eye(3, device=rotvecs.device, dtype=rotvecs.dtype).unsqueeze(0)  # (1, 3, 3)
    R = eye + torch.sin(theta).unsqueeze(-1) * K + (1 - torch.cos(theta).unsqueeze(-1)) * (K @ K)

    return R


def get_phase_shift_grid(
    resolution: int, dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate phase shift grid for translation correction.

    Args:
        resolution: Image resolution
        dtype: Data type for tensors (default: torch.float32)
        device: Device for tensors (default: CPU)

    Returns:
        Tuple of (phase_shift_grid_x, phase_shift_grid_y)
    """
    if device is None:
        device = torch.device("cpu")
    grid_shifted = torch.ceil(torch.arange(-resolution / 2, resolution / 2, dtype=dtype))
    grid_1d = grid_shifted * 2 * np.pi / resolution
    phase_shift_grid_x, phase_shift_grid_y = torch.meshgrid(grid_1d, grid_1d, indexing="xy")

    return phase_shift_grid_x.to(device), phase_shift_grid_y.to(device)


def offset_to_phase_shift(
    offsets: torch.Tensor,
    resolution: Optional[int] = None,
    phase_shift_grid: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Convert translation offsets to phase shift factors.

    Args:
        offsets: Translation offsets of shape (N, 2)
        resolution: Image resolution (optional if phase_shift_grid provided)
        phase_shift_grid: Pre-computed phase shift grid (optional)

    Returns:
        Phase shift factors of shape (N, resolution, resolution)

    Raises:
        ValueError: If neither resolution nor phase_shift_grid is provided
    """
    if phase_shift_grid is None:
        if resolution is None:
            raise ValueError("Either resolution or phase_shift_grid must be provided.")
        phase_shift_grid_x, phase_shift_grid_y = get_phase_shift_grid(resolution)
    else:
        phase_shift_grid_x, phase_shift_grid_y = phase_shift_grid

    phase_shift = torch.exp(
        1j
        * (
            phase_shift_grid_x.unsqueeze(0) * offsets[:, 0].reshape(-1, 1, 1)
            + phase_shift_grid_y.unsqueeze(0) * offsets[:, 1].reshape(-1, 1, 1)
        )
    )

    return phase_shift


class PoseModule(torch.nn.Module):
    """Module for managing particle poses including rotations and translations.

    Attributes:
        resolution: Image resolution
        dtype: Data type for tensors
        use_contrast: Whether to use contrast parameters
        rotvec: Embedding layer for rotation vectors
        offsets: Embedding layer for translation offsets
        contrasts: Embedding layer for contrast parameters (if enabled)
        xy_rot_grid: Grid for rotation calculations
        phase_shift_grid_x: X-component of phase shift grid
        phase_shift_grid_y: Y-component of phase shift grid
    """

    def __init__(
        self,
        init_rotvecs: Union[torch.Tensor, np.ndarray],
        offsets: Union[torch.Tensor, np.ndarray],
        resolution: int,
        dtype: torch.dtype = torch.float32,
        use_contrast: bool = False,
    ) -> None:
        """Initialize PoseModule.

        Args:
            init_rotvecs: Initial rotation vectors of shape (N, 3)
            offsets: Initial translation offsets of shape (N, 2)
            resolution: Image resolution
            dtype: Data type for tensors (default: torch.float32)
            use_contrast: Whether to use contrast parameters (default: False)

        Raises:
            AssertionError: If input shapes are invalid
        """
        super().__init__()
        self.resolution = resolution
        self.dtype = dtype
        self.use_contrast = use_contrast
        # convert to tensor if it's not already
        init_rotvecs = torch.as_tensor(init_rotvecs, dtype=dtype)
        offsets = torch.as_tensor(offsets, dtype=dtype)

        assert init_rotvecs.shape[1] == 3, "Rotation vectors should be of shape (N, 3)"
        assert offsets.shape[1] == 2, "Offsets should be of shape (N, 2)"
        assert (
            init_rotvecs.shape[0] == offsets.shape[0]
        ), "Rotation vectors and offsets should have the same number of elements"

        n = init_rotvecs.shape[0]
        self.rotvec = torch.nn.Embedding(num_embeddings=n, embedding_dim=3, sparse=True)
        self.rotvec.weight.data.copy_(init_rotvecs)
        self.offsets = torch.nn.Embedding(num_embeddings=n, embedding_dim=2, sparse=True, _weight=offsets)
        if self.use_contrast:
            self.contrasts = torch.nn.Embedding(
                num_embeddings=n, embedding_dim=1, sparse=True, _weight=torch.ones((n, 1), dtype=dtype)
            )

        self._init_grid(dtype)

    @property
    def device(self) -> torch.device:
        """Get device of the module.

        Returns:
            Device of the module
        """
        return self.xy_rot_grid.device

    def _init_grid(self, dtype: torch.dtype) -> None:
        """Initialize rotation and phase shift grids.

        Args:
            dtype: Data type for tensors
        """
        grid2d = grid_2d(self.resolution, indexing="yx")
        num_pts = self.resolution**2

        grid = np.pi * np.vstack(
            [
                grid2d["x"].flatten(),
                grid2d["y"].flatten(),
                np.zeros(num_pts, dtype=np.float32),
            ]
        )
        self.xy_rot_grid = torch.tensor(grid, dtype=dtype)

        self.phase_shift_grid_x, self.phase_shift_grid_y = get_phase_shift_grid(self.resolution, dtype=self.dtype)

    def _downsample_grid(self, ds_resolution: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Downsample grids to specified resolution.

        Args:
            ds_resolution: Target resolution

        Returns:
            Tuple of (xy_rot_grid, phase_shift_grid_x, phase_shift_grid_y)
        """
        if ds_resolution == self.resolution:
            return self.xy_rot_grid, self.phase_shift_grid_x, self.phase_shift_grid_y

        xy_rot_grid = crop_image(self.xy_rot_grid.reshape(3, self.resolution, self.resolution), ds_resolution).reshape(
            3, -1
        )
        phase_shift_grid_x = crop_image(self.phase_shift_grid_x, ds_resolution)
        phase_shift_grid_y = crop_image(self.phase_shift_grid_y, ds_resolution)

        return xy_rot_grid, phase_shift_grid_x, phase_shift_grid_y

    def forward(
        self, index: torch.Tensor, ds_resolution: Optional[int] = None
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass to compute rotated points and phase shifts.

        Args:
            index: Indices of particles
            ds_resolution: Downsampled resolution (optional)

        Returns:
            Tuple of (rotated_points, phase_shift) or (rotated_points, phase_shift, contrasts)
        """
        if ds_resolution is None:
            ds_resolution = self.resolution

        xy_rot_grid, phase_shift_grid_x, phase_shift_grid_y = self._downsample_grid(ds_resolution)

        rot_mat = rotvec_to_rotmat(self.rotvec(index))
        pts_rot = torch.flip(
            torch.matmul(rot_mat.reshape(len(index) * 3, 3), xy_rot_grid).reshape(len(index), 3, ds_resolution**2),
            dims=[1],
        )
        pts_rot = (
            torch.remainder(pts_rot + torch.pi, 2 * torch.pi) - torch.pi
        )  # After rotating the grids some of the points can be outside the [-pi , pi]^3 cube

        offsets = -self.offsets(index)
        phase_shift = offset_to_phase_shift(offsets, phase_shift_grid=(phase_shift_grid_x, phase_shift_grid_y))

        if not self.use_contrast:
            return pts_rot, phase_shift
        else:
            return pts_rot, phase_shift, self.contrasts(index)

    def to(self, *args, **kwargs) -> "PoseModule":
        """Move module to specified device.

        Args:
            *args: Positional arguments for torch.nn.Module.to
            **kwargs: Keyword arguments for torch.nn.Module.to

        Returns:
            Self for method chaining
        """
        super().to(*args, **kwargs)
        self.xy_rot_grid = self.xy_rot_grid.to(*args, **kwargs)
        self.phase_shift_grid_x = self.phase_shift_grid_x.to(*args, **kwargs)
        self.phase_shift_grid_y = self.phase_shift_grid_y.to(*args, **kwargs)
        return self

    def get_rotvecs(self) -> torch.Tensor:
        """Get rotation vectors.

        Returns:
            Rotation vectors
        """
        return self.rotvec.weight.data

    def set_rotvecs(self, rotvecs: torch.Tensor, idx: Optional[torch.Tensor] = None) -> None:
        """Set rotation vectors.

        Args:
            rotvecs: New rotation vectors
            idx: Indices to update (optional)
        """
        if idx is None:
            self.rotvec.weight.data.copy_(rotvecs)
        else:
            with torch.no_grad():
                self.rotvec.weight[idx] = rotvecs

    def get_offsets(self) -> torch.Tensor:
        """Get translation offsets.

        Returns:
            Translation offsets
        """
        return self.offsets.weight.data

    def set_offsets(self, offsets: torch.Tensor, idx: Optional[torch.Tensor] = None) -> None:
        """Set translation offsets.

        Args:
            offsets: New translation offsets
            idx: Indices to update (optional)
        """
        if idx is None:
            self.offsets.weight.data.copy_(offsets)
        else:
            with torch.no_grad():
                self.offsets.weight[idx] = offsets

    def get_contrasts(self) -> torch.Tensor:
        """Get contrast parameters.

        Returns:
            Contrast parameters
        """
        return self.contrasts.weight.data

    def set_contrasts(self, contrasts: torch.Tensor, idx: Optional[torch.Tensor] = None) -> None:
        """Set contrast parameters.

        Args:
            contrasts: New contrast parameters
            idx: Indices to update (optional)
        """
        if idx is None:
            self.contrasts.weight.data.copy_(contrasts)
        else:
            with torch.no_grad():
                self.contrasts.weight[idx] = contrasts

    def split_module(self, permutation: Optional[torch.Tensor] = None) -> Tuple["PoseModule", "PoseModule"]:
        """Split module into two modules with non-overlapping subsets of poses.

        Args:
            permutation: Permutation of indices (optional)

        Returns:
            Tuple of (module1, module2)
        """
        n = self.offsets.weight.shape[0]
        device = self.offsets.weight.device
        dtype = self.offsets.weight.dtype

        if permutation is None:
            permutation = torch.arange(n)
        perm = permutation[: n // 2], permutation[n // 2 :]

        # First module: entries at idx
        rotvecs1 = self.rotvec.weight.data[perm[0]].detach().clone()
        offsets1 = self.offsets.weight.data[perm[0]].detach().clone()
        # Second module: entries not in idx
        rotvecs2 = self.rotvec.weight.data[perm[1]].detach().clone()
        offsets2 = self.offsets.weight.data[perm[1]].detach().clone()

        use_contrast = self.use_contrast

        module1 = PoseModule(rotvecs1, offsets1, self.resolution, dtype=dtype, use_contrast=use_contrast)
        module2 = PoseModule(rotvecs2, offsets2, self.resolution, dtype=dtype, use_contrast=use_contrast)

        if use_contrast:
            module1.set_contrasts(self.contrasts.weight.data[perm[0]].detach().clone())
            module2.set_contrasts(self.contrasts.weight.data[perm[1]].detach().clone())

        # Move to same device as original
        module1 = module1.to(device)
        module2 = module2.to(device)

        return module1, module2

    @staticmethod
    def merge_modules(module1: "PoseModule", module2: "PoseModule", permutation: torch.Tensor) -> "PoseModule":
        """Merge two PoseModule instances into a new module.

        Args:
            module1: First module
            module2: Second module
            permutation: Permutation for reordering

        Returns:
            Merged module

        Raises:
            AssertionError: If modules have different contrast settings
        """
        device = module1.rotvec.weight.device
        dtype = module1.rotvec.weight.dtype
        resolution = module1.resolution

        assert module1.use_contrast == module2.use_contrast, "Both modules must have the same value for use_contrast"
        use_contrast = module1.use_contrast

        # Concatenate the weights from both modules
        rotvecs = torch.cat([module1.rotvec.weight.data.detach(), module2.rotvec.weight.data.detach()], dim=0)
        offsets = torch.cat([module1.offsets.weight.data.detach(), module2.offsets.weight.data.detach()], dim=0)

        # Reorder according to permutation
        rotvecs[permutation] = rotvecs.clone()
        offsets[permutation] = offsets.clone()

        merged_module = PoseModule(rotvecs, offsets, resolution, dtype=dtype, use_contrast=use_contrast)

        if use_contrast:
            contrats = torch.cat(
                [module1.contrasts.weight.data.detach(), module2.contrasts.weight.data.detach()], dim=0
            )
            contrats[permutation] = contrats.clone()
            merged_module.set_contrasts(contrats)

        merged_module = merged_module.to(device)
        return merged_module

    def update_from_modules(self, module1: "PoseModule", module2: "PoseModule", permutation: torch.Tensor) -> None:
        """Update module from two sub-modules.

        Args:
            module1: First sub-module
            module2: Second sub-module
            permutation: Permutation for reordering

        Raises:
            AssertionError: If modules have different contrast settings
        """
        assert module1.use_contrast == module2.use_contrast, "Both modules must have the same value for use_contrast"
        use_contrast = module1.use_contrast

        # Concatenate the weights from both modules
        rotvecs = torch.cat([module1.rotvec.weight.data.detach(), module2.rotvec.weight.data.detach()], dim=0)
        offsets = torch.cat([module1.offsets.weight.data.detach(), module2.offsets.weight.data.detach()], dim=0)

        # Reorder according to permutation
        rotvecs[permutation] = rotvecs.clone()
        offsets[permutation] = offsets.clone()

        self.set_rotvecs(rotvecs)
        self.set_offsets(offsets)

        if use_contrast:
            contrats = torch.cat(
                [module1.contrasts.weight.data.detach(), module2.contrasts.weight.data.detach()], dim=0
            )
            contrats[permutation] = contrats.clone()
            self.set_contrasts(contrats)


def estimate_image_offsets_correlation(
    images: torch.Tensor, reference: torch.Tensor, upsampling: int = 4, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Estimate image offsets using cross-correlation with upsampling.

    Args:
        images: Input images of shape (N, L, L)
        reference: Reference image of shape (N, L, L)
        upsampling: Upsampling factor for correlation (default: 4)
        mask: Optional mask to apply (optional)

    Returns:
        Estimated offsets of shape (N, 2)
    """
    n, h, w = images.shape

    images = images * mask if mask is not None else images
    reference = reference * mask if mask is not None else reference
    # Cross-correlation via FFT
    f_img = torch.fft.fft2(images, s=(h * upsampling, w * upsampling))
    f_ref = torch.conj(torch.fft.fft2(reference, s=(h * upsampling, w * upsampling)))
    corr = torch.fft.ifft2(f_img * f_ref).real

    # Center the correlation output
    corr = torch.fft.fftshift(corr)

    max_idx = torch.argmax(corr.reshape(n, -1), dim=1)
    max_idx = torch.unravel_index(max_idx, (h * upsampling, w * upsampling))
    shift_y = max_idx[0] - (h * upsampling) // 2
    shift_x = max_idx[1] - (w * upsampling) // 2

    offsets = -torch.vstack([shift_x, shift_y]).T

    return offsets


def estimate_image_offsets_newton(
    images: torch.Tensor,
    reference: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    init_offsets: Optional[torch.Tensor] = None,
    in_fourier_domain: bool = False,
    obj_func: Optional[callable] = None,
) -> torch.Tensor:
    """Estimate image offsets using Newton's method optimization.

    Args:
        images: Input images
        reference: Reference image
        mask: Optional mask to apply (optional)
        init_offsets: Initial offset estimates (optional)
        in_fourier_domain: Whether input is in Fourier domain (default: False)
        obj_func: Custom objective function (optional)

    Returns:
        Estimated offsets
    """
    if not in_fourier_domain:
        dtype = images.dtype
        images = centered_fft2(images * mask) if mask is not None else centered_fft2(images)
        reference = centered_fft2(reference * mask) if mask is not None else centered_fft2(reference)
    else:
        dtype = images.real.dtype
        images = centered_fft2(centered_ifft2(images).real * mask) if mask is not None else images
        reference = centered_fft2(centered_ifft2(reference).real * mask) if mask is not None else reference

    if init_offsets is None:
        init_offsets = torch.zeros((images.shape[0], 2), dtype=dtype, device=images.device)

    phase_shift_grid = get_phase_shift_grid(images.shape[-1], dtype=dtype, device=images.device)

    if obj_func is None:

        def obj_func(phase_shifted_image):
            return torch.norm(phase_shifted_image - reference, dim=(-1, -2)) ** 2

    init_offsets.requires_grad = True

    optimizer = BlockNewtonOptimizer(
        [init_offsets], beta=0.2, c=0, line_search=True, max_ls_steps=3, step_size_limit=0.5
    )

    for _ in range(10):

        def closure():
            optimizer.zero_grad()
            phase_shifted_image = images * offset_to_phase_shift(-init_offsets, phase_shift_grid=phase_shift_grid)
            loss = obj_func(phase_shifted_image)

            return loss

        optimizer.step(closure)
        optimizer.zero_grad()

    init_offsets.requires_grad = False

    return init_offsets


def estimate_image_offsets(
    images: torch.Tensor, reference: torch.Tensor, mask: Optional[torch.Tensor] = None, in_fourier_domain: bool = False
) -> torch.Tensor:
    """Estimate image offsets using correlation followed by Newton optimization.

    Args:
        images: Input images
        reference: Reference image
        mask: Optional mask to apply (optional)
        in_fourier_domain: Whether input is in Fourier domain (default: False)

    Returns:
        Estimated offsets
    """
    if not in_fourier_domain:
        images = centered_fft2(images * mask) if mask is not None else centered_fft2(images)
        reference = centered_fft2(reference * mask) if mask is not None else centered_fft2(reference)
    else:
        images = centered_fft2(centered_ifft2(images).real * mask) if mask is not None else images
        reference = centered_fft2(centered_ifft2(reference).real * mask) if mask is not None else reference

    corr = torch.fft.ifft2(images * torch.conj(reference)).real

    # Center the correlation output
    corr = torch.fft.fftshift(corr)

    max_idx = torch.argmax(corr.reshape(images.shape[0], -1), dim=1)
    max_idx = torch.unravel_index(max_idx, images.shape[-2:])
    shift_y = max_idx[0] - (images.shape[-2]) // 2
    shift_x = max_idx[1] - (images.shape[-1]) // 2

    init_offsets = -torch.vstack([shift_x, shift_y]).T.to(images.real.dtype)

    return estimate_image_offsets_newton(images, reference, init_offsets=init_offsets, in_fourier_domain=True)


def find_global_alignment(rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
    """Find the best global rotation Q such that rot1 ≈ Q @ rot2.

    Uses orthogonal Procrustes via SVD.
    """
    # rot1, rot2: (N, 3, 3)
    M = torch.zeros((3, 3), dtype=rot1.dtype, device=rot1.device)
    for i in range(rot1.shape[0]):
        M += rot1[i] @ rot2[i].T

    U, _, Vt = torch.linalg.svd(M)
    Q = U @ Vt

    # Ensure det(Q) = +1 (proper rotation)
    if torch.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    return Q


def out_of_plane_rot_error(
    rot1: torch.Tensor, rot2: torch.Tensor, global_align: bool = False
) -> Tuple[np.ndarray, float, float]:
    """Compute out-of-plane rotation error between two sets of rotation matrices.

    Implementation is used from DRGN-AI
    https://github.com/ml-struct-bio/drgnai/blob/d45341d1f3411d6db6da6f557207f10efd16da17/src/metrics.py#L134

    Args:
        rot1: First set of rotation matrices
        rot2: Second set of rotation matrices
        global_align: Whether to perform global alignment of rotations before computing error

    Returns:
        Tuple of (angles, mean_angle, median_angle) in degrees
    """
    if global_align:
        Q = find_global_alignment(rot1, rot2)
        rot2 = torch.einsum("ij,njk->nik", Q, rot2)  # rot2' = Q @ rot2

    unitvec_gt = torch.tensor([0, 0, 1], dtype=torch.float32).reshape(3, 1)

    out_of_planes_1 = torch.sum(rot1 * unitvec_gt, dim=-2)
    out_of_planes_1 = out_of_planes_1.numpy()
    out_of_planes_1 /= np.linalg.norm(out_of_planes_1, axis=-1, keepdims=True)

    out_of_planes_2 = torch.sum(rot2 * unitvec_gt, dim=-2)
    out_of_planes_2 = out_of_planes_2.numpy()
    out_of_planes_2 /= np.linalg.norm(out_of_planes_2, axis=-1, keepdims=True)

    angles = np.arccos(np.clip(np.sum(out_of_planes_1 * out_of_planes_2, -1), -1.0, 1.0)) * 180.0 / np.pi

    return angles, np.mean(angles), np.median(angles)


def in_plane_rot_error(
    rot1: Union[torch.Tensor, np.ndarray], rot2: Union[torch.Tensor, np.ndarray], global_align: bool = False
) -> Tuple[np.ndarray, float, float]:
    """Compute the in-plane rotation error (in degrees) between two sets of rotation matrices.

    The in-plane rotation is the rotation about the z-axis (beam axis).

    Args:
        rot1: First set of rotation matrices
        rot2: Second set of rotation matrices
        global_align: Whether to perform global alignment of rotations before computing error

    Returns:
        Tuple of (angles, mean_angle, median_angle) in degrees
    """
    if global_align:
        Q = find_global_alignment(rot1, rot2)
        rot2 = torch.einsum("ij,njk->nik", Q, rot2)  # rot2' = Q @ rot2
    # The in-plane rotation angle psi can be extracted from the rotation matrix as:
    # psi = atan2(R[1,0], R[0,0])
    # rot1, rot2: (..., 3, 3) tensors

    # Ensure input is torch tensor
    if not torch.is_tensor(rot1):
        rot1 = torch.tensor(rot1)
    if not torch.is_tensor(rot2):
        rot2 = torch.tensor(rot2)

    # Extract psi angles for each rotation
    psi1 = torch.atan2(rot1[..., 1, 0], rot1[..., 0, 0])
    psi2 = torch.atan2(rot2[..., 1, 0], rot2[..., 0, 0])

    # Compute difference, wrap to [-pi, pi]
    dpsi = psi1 - psi2
    dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi

    angles = torch.abs(dpsi) * 180.0 / np.pi
    angles = angles.cpu().numpy()

    mean_angle = np.mean(angles)
    median_angle = np.median(angles)

    return angles, mean_angle, median_angle


def offset_mean_error(offsets1: torch.Tensor, offsets2: torch.Tensor, L: Optional[int] = None) -> float:
    """Compute mean error between two sets of offsets.

    Args:
        offsets1: First set of offsets
        offsets2: Second set of offsets
        L: Image resolution for normalization (optional)

    Returns:
        Mean error (normalized by L if provided)
    """
    mean_err = torch.norm(offsets1 - offsets2, dim=1).mean()
    if L is not None:
        mean_err /= L
    return mean_err.item()
