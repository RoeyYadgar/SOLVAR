import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from aspire.utils import grid_3d

from solvar.nufft_plan import BaseNufftPlan, NufftSpec, nufft_adjoint, nufft_forward


def pad_tensor(tensor: torch.Tensor, size: List[int], dims: Optional[List[int]] = None) -> torch.Tensor:
    """Pad tensor to specified size along given dimensions.

    Args:
        tensor: Input tensor to pad
        size: Target size for each dimension
        dims: Dimensions to pad (default: last N dimensions)

    Returns:
        Padded tensor
    """
    tensor_shape = tensor.shape
    if dims is None:
        dims = [(-1 - i) % tensor.ndim for i in range(len(size))]
    padded_tensor_size = torch.tensor(tensor_shape)
    padded_tensor_size[dims] = torch.tensor(size)
    padded_tensor = torch.zeros(list(padded_tensor_size), dtype=tensor.dtype, device=tensor.device)

    num_dims = len(size)
    start_ind = [math.floor(tensor_shape[dims[i]] / 2) - math.floor(size[i] / 2) for i in range(num_dims)]

    slice_ind = tuple([slice(-start_ind[i], tensor_shape[dims[i]] - start_ind[i]) for i in range(num_dims)])
    slice_ind_full = [slice(tensor.shape[i]) for i in range(tensor.ndim)]
    for i in range(num_dims):
        slice_ind_full[dims[i]] = slice_ind[i]
    padded_tensor[slice_ind_full] = tensor

    return padded_tensor


def crop_tensor(tensor: torch.Tensor, size: List[int], dims: Optional[List[int]] = None) -> torch.Tensor:
    """Crop tensor to specified size along given dimensions.

    Args:
        tensor: Input tensor to crop
        size: Target size for each dimension
        dims: Dimensions to crop (default: last N dimensions)

    Returns:
        Cropped tensor
    """
    tensor_shape = tensor.shape
    if dims is None:
        dims = [(-1 - i) % tensor.ndim for i in range(len(size))]

    num_dims = len(size)
    start_ind = [math.floor(tensor_shape[dims[i]] / 2) - math.floor(size[i] / 2) for i in range(num_dims)]

    slice_ind = tuple([slice(start_ind[i], size[i] + start_ind[i]) for i in range(num_dims)])
    slice_ind_full = [slice(tensor.shape[i]) for i in range(tensor.ndim)]
    for i in range(num_dims):
        slice_ind_full[dims[i]] = slice_ind[i]

    return tensor[slice_ind_full]


def crop_image(images: torch.Tensor, L_crop: int) -> torch.Tensor:
    """Crop images to specified size.

    Args:
        images: Input images
        L_crop: Target crop size

    Returns:
        Cropped images
    """
    L = images.shape[-1]
    img_idx = torch.arange(-(L_crop // 2), L_crop // 2 + L_crop % 2) + L // 2
    return images[..., img_idx, :][..., img_idx].reshape(*images.shape[:-2], L_crop, L_crop)


def centered_fft2(
    image: torch.Tensor, im_dim: List[int] = [-1, -2], padding_size: Optional[List[int]] = None
) -> torch.Tensor:
    """Compute centered 2D FFT.

    Args:
        image: Input image
        im_dim: Dimensions to apply FFT to (default: [-1, -2])
        padding_size: Size to pad to before FFT (optional)

    Returns:
        Centered 2D FFT
    """
    return _centered_fft(torch.fft.fft2, image, im_dim, padding_size)


def centered_ifft2(
    image: torch.Tensor, im_dim: List[int] = [-1, -2], cropping_size: Optional[List[int]] = None
) -> torch.Tensor:
    """Compute centered 2D inverse FFT.

    Args:
        image: Input image
        im_dim: Dimensions to apply IFFT to (default: [-1, -2])
        cropping_size: Size to crop to after IFFT (optional)

    Returns:
        Centered 2D inverse FFT
    """
    tensor = _centered_fft(torch.fft.ifft2, image, im_dim)
    return crop_tensor(tensor, cropping_size, im_dim) if cropping_size is not None else tensor


def centered_fft3(
    image: torch.Tensor, im_dim: List[int] = [-1, -2, -3], padding_size: Optional[List[int]] = None
) -> torch.Tensor:
    """Compute centered 3D FFT.

    Args:
        image: Input image
        im_dim: Dimensions to apply FFT to (default: [-1, -2, -3])
        padding_size: Size to pad to before FFT (optional)

    Returns:
        Centered 3D FFT
    """
    return _centered_fft(torch.fft.fftn, image, im_dim, padding_size)


def centered_ifft3(
    image: torch.Tensor, im_dim: List[int] = [-1, -2, -3], cropping_size: Optional[List[int]] = None
) -> torch.Tensor:
    """Compute centered 3D inverse FFT.

    Args:
        image: Input image
        im_dim: Dimensions to apply IFFT to (default: [-1, -2, -3])
        cropping_size: Size to crop to after IFFT (optional)

    Returns:
        Centered 3D inverse FFT
    """
    tensor = _centered_fft(torch.fft.ifftn, image, im_dim)
    return crop_tensor(tensor, cropping_size, im_dim) if cropping_size is not None else tensor


def _centered_fft(
    fft_func, tensor: torch.Tensor, dim: List[int], size: Optional[List[int]] = None, **fft_kwargs
) -> torch.Tensor:
    """Helper function for centered FFT operations.

    Args:
        fft_func: FFT function to apply
        tensor: Input tensor
        dim: Dimensions to apply FFT to
        size: Size to pad to before FFT (optional)
        **fft_kwargs: Additional FFT arguments

    Returns:
        Centered FFT result
    """
    if size is not None:
        tensor = pad_tensor(tensor, size, dim)
    return torch.fft.fftshift(fft_func(torch.fft.ifftshift(tensor, dim=dim, **fft_kwargs), dim=dim), dim=dim)


def make_nufft_plan(
    spec: NufftSpec, input: Optional[torch.Tensor] = None
) -> Union[BaseNufftPlan, Tuple[BaseNufftPlan, torch.Tensor]]:
    """Create and configure a NUFFT (Non-uniform Fast Fourier Transform) plan from a NufftSpec.

    Args:
        spec (NufftSpec): The NUFFT specification containing configuration parameters, including
            the NUFFT plan type and arguments.
        input (Optional[torch.Tensor]): Optional input tensor in spatial domain. If provided,
            the input is transformed into the correct domain the nufft plan expects.

    Returns:
        BaseNufftPlan or Tuple[BaseNufftPlan, torch.Tensor]:
            If `input` is None, returns the constructed NUFFT plan.
            If `input` is provided, returns a tuple of the constructed NUFFT plan and the (possibly
            transformed) input tensor.
    """
    # TODO: Ideally this should be implemented in nufft_plan.py
    # But that causes circular import errors - fix it.

    nufft_plan = spec.nufft_type(**spec.get_nufft_kwargs())

    if input is None:
        return nufft_plan

    if spec.input_in_fourier:
        input = centered_fft3(input, padding_size=tuple(v * spec.upsample_factor for v in input.shape[-3:]))

    return nufft_plan, input


def preprocess_image_batch(
    images: torch.Tensor,
    nufft_plan: BaseNufftPlan,
    filters: torch.Tensor,
    pose: Tuple[torch.Tensor, torch.Tensor],
    mean_volume: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    mask_threshold: Optional[float] = None,
    softening_kernel_fourier: Optional[torch.Tensor] = None,
    fourier_domain: bool = False,
) -> torch.Tensor:
    """Preprocess image batch by shifting images, subtracting projected mean volume and applying
    masking.

    Args:
        images: Input images
        nufft_plan: NUFFT plan for projection
        filters: CTF filters
        pose: Tuple of (rotated points, phase shift)
        mean_volume: Mean volume to subtract
        mask: Volume mask (optional)
        mask_threshold: Threshold for mask (optional)
        softening_kernel_fourier: Softening kernel in Fourier domain (optional)
        fourier_domain: Whether to return in Fourier domain (default: False)

    Returns:
        Preprocessed images
    """
    pts_rot, phase_shift = pose
    nufft_plan.setpts(pts_rot)

    if not fourier_domain:
        images = centered_fft2(images)

    images = images * phase_shift

    mean_forward = vol_forward(mean_volume, nufft_plan, filters=filters, fourier_domain=True).squeeze(1)
    images = images - mean_forward

    if mask is not None:
        images = centered_ifft2(images).real
        mask_forward = (
            vol_forward(mask, nufft_plan, filters=None, fourier_domain=False).squeeze(1).detach()
        )  # We don't want to take the gradient with respect to the mask (in case the pose is being optimized)
        mask_forward = mask_forward > mask_threshold
        soft_mask = centered_ifft2(centered_fft2(mask_forward) * softening_kernel_fourier).real
        images *= soft_mask

        if fourier_domain:
            images = centered_fft2(images)
    elif not fourier_domain:
        images = centered_ifft2(images).real

    return images


def get_mask_threshold(mask: torch.Tensor, nufft_plan: BaseNufftPlan) -> float:
    """Get threshold for mask projection.

    Args:
        mask: Volume mask
        nufft_plan: NUFFT plan for projection

    Returns:
        Mask threshold value
    """
    projected_mask = vol_forward(mask, nufft_plan).squeeze(1)
    vals = projected_mask.reshape(-1).cpu().numpy()
    return np.percentile(
        vals[vals > 10 ** (-1.5)], 10
    )  # filter values which aren't too close to 0 and take a threhosld that captures 90% of the projected mask


def lowpass_volume(volume: torch.Tensor, cutoff: float, lowpass_shape: str = "rect") -> torch.Tensor:
    """Apply low-pass filter to volume.

    Args:
        volume: Input volume
        cutoff: Cutoff frequency
        lowpass_shape: Shape of filter ("rect" or "sphere") (default: "rect")

    Returns:
        Low-pass filtered volume
    """
    fourier_vol = centered_fft3(volume)
    L = volume.shape[-1]
    if lowpass_shape == "rect":
        fourier_mask = torch.arange(-L // 2, L // 2) if L % 2 == 0 else torch.arange(-L // 2, L // 2) + 1
        fourier_mask = torch.abs(fourier_mask.to(volume.device)) < cutoff
        fourier_mask = torch.einsum("i,j,k->ijk", fourier_mask, fourier_mask, fourier_mask)
    elif lowpass_shape == "sphere":
        fourier_mask = torch.tensor(grid_3d(L, normalized=False)["r"], device=volume.device)
        fourier_mask = torch.abs(fourier_mask) < cutoff

    fourier_vol *= fourier_mask.unsqueeze(0)
    return centered_ifft3(fourier_vol).real


def highpass_volume(volume: torch.Tensor, cutoff: float, highpass_shape: str = "rect") -> torch.Tensor:
    """Apply high-pass filter to volume.

    Args:
        volume: Input volume
        cutoff: Cutoff frequency
        highpass_shape: Shape of filter ("rect" or "sphere") (default: "rect")

    Returns:
        High-pass filtered volume
    """
    fourier_vol = centered_fft3(volume)
    L = volume.shape[-1]
    if highpass_shape == "rect":
        fourier_mask = torch.arange(-L // 2, L // 2) if L % 2 == 0 else torch.arange(-L // 2, L // 2) + 1
        fourier_mask = torch.abs(fourier_mask.to(volume.device)) > cutoff
        fourier_mask = torch.einsum("i,j,k->ijk", fourier_mask, fourier_mask, fourier_mask)
    elif highpass_shape == "sphere":
        fourier_mask = torch.tensor(grid_3d(L, normalized=False)["r"], device=volume.device)
        fourier_mask = torch.abs(fourier_mask) > cutoff
    fourier_vol *= fourier_mask.unsqueeze(0)
    return centered_ifft3(fourier_vol).real


def vol_forward(
    volume: torch.Tensor,
    plan: Union[BaseNufftPlan, List[BaseNufftPlan]],
    filters: Optional[torch.Tensor] = None,
    fourier_domain: bool = False,
) -> torch.Tensor:
    """Forward project volume to images.

    Args:
        volume: Input volume
        plan: NUFFT plan or list of plans
        filters: CTF filters (optional)
        fourier_domain: Whether to return in Fourier domain (default: False)

    Returns:
        Projected images
    """
    L = plan.sz[-1]
    if isinstance(plan, (list, tuple)):  # When multiple plans are given loop through them
        volume_forward = torch.zeros((len(plan), volume.shape[0], L, L), dtype=volume.dtype, device=volume.device)
        for i in range(len(plan)):
            volume_forward[i] = (
                vol_forward(volume, plan[i], filters[i]) if filters is not None else vol_forward(volume, plan[i])
            )
        return volume_forward
    elif isinstance(plan, BaseNufftPlan):
        vol_nufft = nufft_forward(volume, plan)
        vol_nufft = (
            vol_nufft.reshape((*volume.shape[:-3], -1, vol_nufft.shape[-2], vol_nufft.shape[-1]))
            .transpose(0, 1)
            .clone()
        )
        batch_size = vol_nufft.shape[1]

        if vol_nufft.shape[-1] % 2 == 0:
            # vol_nufft_clone = vol_nufft.clone()
            vol_nufft[:, :, 0, :] = 0
            vol_nufft[:, :, :, 0] = 0
        else:
            vol_nufft = vol_nufft

        if filters is not None:
            vol_nufft = vol_nufft * filters.unsqueeze(1)

        if batch_size == 1:
            vol_nufft = vol_nufft.squeeze(0)

        volume_forward = centered_ifft2(vol_nufft).real if (not fourier_domain) else vol_nufft

        return volume_forward / L


def im_backward(
    image: torch.Tensor, plan: BaseNufftPlan, filters: Optional[torch.Tensor] = None, fourier_domain: bool = False
) -> torch.Tensor:
    """Backward project images to volume.

    Args:
        image: Input images
        plan: NUFFT plan
        filters: CTF filters (optional)
        fourier_domain: Whether input is in Fourier domain (default: False)

    Returns:
        Back-projected volume
    """
    L = image.shape[-1]
    im_fft = centered_fft2(image / L**2) if (not fourier_domain) else image.clone()

    if filters is not None:
        im_fft *= filters

    if L % 2 == 0:
        im_fft[:, 0, :] = 0
        im_fft[:, :, 0] = 0

    image_backward = nufft_adjoint(im_fft, plan)

    return torch.real(image_backward) / L if (not fourier_domain) else image_backward / L
