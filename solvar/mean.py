import logging
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from aspire.utils import grid_3d
from torch.utils.data import DataLoader
from tqdm import tqdm

from solvar.dataset import CovarDataset, LazyCovarDataset, create_dataloader, get_ordered_half_split
from solvar.fsc_utils import (
    average_fourier_shell,
    expand_fourier_shell,
    rpsd,
    upsample_and_expand_fourier_shell,
    vol_fsc,
)
from solvar.nufft_plan import NufftPlanDiscretized
from solvar.poses import get_phase_shift_grid, offset_to_phase_shift
from solvar.projection_funcs import centered_fft3, centered_ifft3, im_backward
from solvar.source import ImageSource
from solvar.utils import get_complex_real_dtype, get_cpu_count, get_torch_device

logger = logging.getLogger(__name__)


def get_grid_correction(L: int, upsampling_factor: int, nufft_disc: str):
    assert nufft_disc in ("bilinear", "nearest"), "nufft_disc must be 'bilinear' or 'nearest'"

    pixel_pos = torch.arange(-(L // 2), (L - 1) // 2 + 1) / L
    pixel_pos = torch.pi * pixel_pos / upsampling_factor
    sinc_val = torch.sin(pixel_pos) / pixel_pos
    sinc_val[pixel_pos == 0] = 1
    sinc_val[sinc_val < 1e-6] = 1

    if nufft_disc == "bilinear":
        sinc_val = sinc_val**2

    sinc_volume = torch.einsum("i,j,k->ijk", sinc_val, sinc_val, sinc_val)
    return sinc_volume


def complex_conjugate_gradient(A_mv, b, x0=None, rtol=1e-6, maxiter=None):
    """Solves the linear system Ax = b for complex-valued inputs using the Conjugate Gradient
    method.

    Args:
        A_mv (callable): A function that computes the matrix-vector product A @ v.
        b (torch.Tensor): The right-hand side vector (complex dtype expected).
        x0 (torch.Tensor, optional): Initial guess for the solution. Defaults to None (zero vector).
        rtol (float, optional): Relative tolerance for convergence (norm(r) < rtol * norm(b)).
        maxiter (int, optional): Maximum number of iterations. Defaults to 10 * len(b).

    Returns:
        torch.Tensor: The approximate solution vector x.
    """
    if not b.is_complex():
        raise ValueError("Input vector b must be a complex tensor.")

    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - A_mv(x)
    p = r.clone()
    # Use torch.vdot for complex inner product (conjugates the first argument)
    rs_old = torch.vdot(r, r).real  # The result should always be real

    if maxiter is None:
        maxiter = len(b) * 10

    b_norm = torch.norm(b)

    for num_iter in range(maxiter):
        Ap = A_mv(p)
        # alpha = rs_old / torch.vdot(p, Ap)
        p_Ap_vdot = torch.vdot(p, Ap)
        alpha = rs_old / p_Ap_vdot  # alpha is a scalar (potentially complex, but ideally real for HPD A)

        x += alpha * p
        r -= alpha * Ap

        rs_new = torch.vdot(r, r).real

        # Check for convergence using the norm of the residual
        if torch.sqrt(rs_new) < rtol * b_norm:
            break

        # Update search direction
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    logger.debug(f"CG converged in {num_iter} iterations")

    return x


def correct_upsampling_padding(
    lhs: torch.Tensor, rhs: torch.Tensor, cg_init: torch.Tensor, L: int, upsampling_factor: int
) -> torch.tensor:
    """Corrects for non-diagonal least-squares operator caused by zero padding in spatial domain
    (upsampling in Fourier).

    Args:
        lhs (torch.Tensor): The left-hand side operator (in Fourier space,
            shape (L*upsampling_factor, L*upsampling_factor, L*upsampling_factor)).
        rhs (torch.Tensor): The right-hand side vector (in Fourier space, shape matching lhs).
        cg_init (torch.Tensor): Initial guess for the iterative CG solver (in spatial domain, shape (L,L,L)).
        L (int): Original volume edge length (before upsampling).
        upsampling_factor (int): Upsampling factor applied to volume in Fourier space.

    Returns:
        torch.Tensor: The corrected mean volume tensor (in spatial domain, shape (L, L, L)).
    """
    # When upsampling, the least squares operator will not be diagonal exactly in Fourier space
    # this is because it is of the form (\sum F^{*}_us M^T F^{*} P_i^T P_i F M F_us)
    # where F, F_us are the fourier transform in original and upsampled size, M is the zero padding operator.
    # To correct for this we perform CG iterations

    def A_operator(x):
        x = x.reshape((L, L, L))
        x_new = centered_fft3(x, padding_size=(upsampling_factor * L,) * 3)
        x_new = lhs * x_new
        x_new = centered_ifft3(x_new, cropping_size=(L,) * 3)
        return x_new.reshape(-1)

    mean_volume = complex_conjugate_gradient(
        A_operator,
        b=centered_ifft3(rhs, cropping_size=(L,) * 3).reshape(-1),
        x0=torch.complex(cg_init.reshape(-1), torch.zeros_like(cg_init.reshape(-1))),
    )
    mean_volume = mean_volume.reshape((L, L, L)).real

    return mean_volume


def regularize_lhs(lhs: torch.Tensor) -> torch.Tensor:
    reg = average_fourier_shell(lhs.real) * 1e-3
    reg = expand_fourier_shell(reg, lhs.shape[-1], 3)

    reg_lhs = torch.max(lhs.abs(), reg)
    reg_lhs = torch.clamp(reg_lhs, min=1e-8)

    return reg_lhs


def reconstruct_mean(
    dataset: Union[CovarDataset, DataLoader],
    init_vol: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    upsampling_factor: int = 2,
    batch_size: int = 1024,
    idx: Optional[torch.Tensor] = None,
    do_grid_correction: Optional[bool] = True,
    return_lhs_rhs: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Reconstruct mean volume from particle images.

    Args:
        dataset: Dataset or DataLoader containing particle images
        init_vol: Initial volume for regularization (optional)
        mask: Volume mask to apply (optional)
        upsampling_factor: Upsampling factor for NUFFT (default: 2)
        batch_size: Batch size for processing (default: 1024)
        idx: Indices of particles to use (optional)
        do_grid_correction: Whether to perform grid correction of Nufft Discretization (default: True)
        return_lhs_rhs: Whether to return left-hand side and right-hand side terms

    Returns:
        Reconstructed mean volume or tuple of (mean_volume, lhs, rhs)
    """
    if not isinstance(dataset, DataLoader):
        if idx is None:
            idx = torch.arange(len(dataset))
        num_workers = min(4, get_cpu_count() - 1)
        dataloader = create_dataloader(
            dataset, batch_size=batch_size, idx=idx, pin_memory=True, num_workers=num_workers
        )
    else:
        dataloader = dataset
        assert idx is None, "If input dataset is a dataloader, idx cannot be specified"
    dataset = dataloader.dataset

    device = get_torch_device() if init_vol is None else init_vol.device

    L = dataset.resolution
    nufft_plan = NufftPlanDiscretized((L,) * 3, upsample_factor=upsampling_factor, mode="nearest", use_half_grid=False)

    is_dataset_in_fourier = not dataset._in_spatial_domain

    if not is_dataset_in_fourier:
        dataset.to_fourier_domain()

    complex_dtype = get_complex_real_dtype(dataset.dtype)
    backproj_im = torch.zeros((L * upsampling_factor,) * 3, device=device, dtype=complex_dtype)
    backproj_ctf = torch.zeros((L * upsampling_factor,) * 3, device=device, dtype=complex_dtype)
    phase_shift_grid = get_phase_shift_grid(L, dtype=backproj_im.real.dtype, device=device)

    for batch in tqdm(dataloader, desc="Reconstructing mean volume"):
        images, pts_rot, filters, idx = batch
        image_offsets = dataset.offsets[idx].to(device).to(pts_rot.dtype)
        image_contrasts = dataset.contrasts[idx].to(device)
        images = images.to(device) * offset_to_phase_shift(-image_offsets, phase_shift_grid=phase_shift_grid)
        pts_rot = pts_rot.to(device)
        filters = filters.to(device)
        filters *= image_contrasts.reshape(-1, 1, 1)

        nufft_plan.setpts(pts_rot)

        backproj_im += im_backward(images, nufft_plan, filters, fourier_domain=True)[0]
        backproj_ctf += im_backward(
            torch.complex(filters, torch.zeros_like(filters)), nufft_plan, filters, fourier_domain=True
        )[0]

    # normalization by L is needed because backproj_ctf represnts diag(\sum P^T P)
    # and the projection operator and since we only use P^T we are not taking into
    # account a division by L in vol_forward
    backproj_ctf /= L

    if not is_dataset_in_fourier:
        dataset.to_spatial_domain()

    if init_vol is not None:
        init_vol_rpsd = rpsd(init_vol.squeeze(0))
        reg = dataset.noise_var / upsample_and_expand_fourier_shell(init_vol_rpsd, L * upsampling_factor, 3)
        reg /= L
        from matplotlib import pyplot as plt

        v = average_fourier_shell(reg, backproj_ctf)
        fig = plt.figure()
        plt.plot(v.cpu().T)
        plt.yscale("log")
        fig.savefig("test.jpg")

        reg_backproj_ctf = backproj_ctf + reg
    else:
        # reg = backproj_ctf.abs().max() * 1e-5
        reg_backproj_ctf = regularize_lhs(backproj_ctf)

    mean_volume = backproj_im / reg_backproj_ctf
    mean_volume = centered_ifft3(mean_volume, cropping_size=(L,) * 3).real

    if upsampling_factor > 1:
        mean_volume = correct_upsampling_padding(reg_backproj_ctf, backproj_im, mean_volume, L, upsampling_factor)

    # Zero out frequencies outside the sphere
    mean_volume = centered_ifft3(
        centered_fft3(mean_volume) * torch.tensor(grid_3d(L, shifted=False, normalized=True)["r"] <= 1, device=device)
    ).real

    if do_grid_correction:
        grid_correction = get_grid_correction(L, upsampling_factor, "nearest").to(device)
        mean_volume = mean_volume / grid_correction

    if mask is not None:
        mean_volume *= mask.squeeze(0)

    if not return_lhs_rhs:
        return mean_volume
    else:
        return mean_volume, backproj_im, backproj_ctf


def reconstruct_mean_from_halfsets(
    dataset: CovarDataset, idx: Optional[torch.Tensor] = None, **reconstruction_kwargs: Any
) -> torch.Tensor:
    """Reconstruct mean volume using half-sets for regularization.

    Args:
        dataset: Dataset containing particle images
        idx: Indices of particles to use (optional)
        **reconstruction_kwargs: Additional arguments for reconstruction

    Returns:
        Regularized mean volume
    """
    reconstruction_kwargs["return_lhs_rhs"] = True
    if idx is None:
        idx = torch.concatenate(get_ordered_half_split(len(dataset)))

    mean_half1, rhs1, lhs1 = reconstruct_mean(dataset, idx=idx[: len(idx) // 2], **reconstruction_kwargs)
    mean_half2, rhs2, lhs2 = reconstruct_mean(dataset, idx=idx[len(idx) // 2 :], **reconstruction_kwargs)

    return regularize_mean_from_halfsets(
        mean_half1,
        rhs1,
        lhs1,
        mean_half2,
        rhs2,
        lhs2,
        mask=reconstruction_kwargs.get("mask", None),
        do_grid_correction=reconstruction_kwargs.get("do_grid_correction", True),
    )


def reconstruct_mean_from_halfsets_DDP(
    dataset: DataLoader, ranks: Optional[List[int]] = None, **reconstruction_kwargs: Any
) -> torch.Tensor:
    """Reconstruct mean volume using half-sets with distributed data parallel processing.

    This function assumes the input dataloader has a distributed sampler.
    Each node will only pass on its corresponding samples determined by the sampler.

    Args:
        dataset: DataLoader with distributed sampler
        ranks: List of ranks to use (default: all available ranks)
        **reconstruction_kwargs: Additional arguments for reconstruction

    Returns:
        Regularized mean volume
    """
    if ranks is None:
        ranks = [i for i in range(dist.get_world_size())]

    if len(ranks) == 1:
        # In the case there's only one rank, we call the non DDP version using the internal dataset of the dataloader
        # and idx selected by the sampler
        return reconstruct_mean_from_halfsets(dataset.dataset, idx=list(iter(dataset.sampler)), **reconstruction_kwargs)
    reconstruction_kwargs["return_lhs_rhs"] = True

    world_size = len(ranks)
    rank = dist.get_rank()

    result = reconstruct_mean(dataset, **reconstruction_kwargs)
    mean, backproj_im, backproj_ctf = result

    # Sum backproj_im and backproj_ctf only on the group of the corresponding half set
    group1 = dist.new_group(ranks=ranks[: world_size // 2])
    group2 = dist.new_group(ranks=ranks[world_size // 2 :])
    rank_group = group1 if rank in ranks[: world_size // 2] else group2
    dist.all_reduce(backproj_im, op=dist.ReduceOp.SUM, group=rank_group)
    dist.all_reduce(backproj_ctf, op=dist.ReduceOp.SUM, group=rank_group)

    mean_volume = backproj_im / (backproj_ctf + 1e-1)
    mean_volume = centered_ifft3(mean_volume, cropping_size=(mean.shape[-1],) * 3).real

    # Sync mean_volume,backproj_im,backproj_ctf across the two groups
    half1 = []
    half2 = []
    for i, tensor in enumerate([mean_volume, backproj_im, backproj_ctf]):
        # TODO: this is inefficent since tensor is the same across each group.
        # This can be done instead by sending an receiveing the tensor for rank pairs from the two groups
        tensor_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        tensor = tensor.contiguous()
        dist.all_gather(tensor_list, tensor)

        half1.append(tensor_list[ranks[0]])
        half2.append(tensor_list[ranks[-1]])

    dist.destroy_process_group(group1)
    dist.destroy_process_group(group2)

    return regularize_mean_from_halfsets(
        *half1,
        *half2,
        reconstruction_kwargs.get("mask", None),
        do_grid_correction=reconstruction_kwargs.get("do_grid_correction", True),
    )


def regularize_mean_from_halfsets(
    mean_half1: torch.Tensor,
    rhs1: torch.Tensor,
    lhs1: torch.Tensor,
    mean_half2: torch.Tensor,
    rhs2: torch.Tensor,
    lhs2: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    do_grid_correction: Optional[bool] = True,
) -> torch.Tensor:
    """Regularize mean volume using FSC-based regularization from half-sets.

    Args:
        mean_half1: First half-set mean volume
        rhs1: First half-set left-hand side term
        lhs1: First half-set right-hand side term
        mean_half2: Second half-set mean volume
        rhs2: Second half-set left-hand side term
        lhs2: Second half-set right-hand side term
        mask: Volume mask to apply (optional)
        do_grid_correction: Whether to perform grid correction of Nufft Discretization (default: True)

    Returns:
        Regularized mean volume
    """
    L = mean_half1.shape[-1]
    upsampling_factor = lhs1.shape[-1] // L

    filter_gain = centered_fft3(centered_ifft3((lhs1 + lhs2), cropping_size=(L,) * 3)).abs() / 2

    averaged_filter_gain = average_fourier_shell(1 / filter_gain).to(mean_half1.device)

    mean_fsc = vol_fsc(mean_half1, mean_half2)
    fsc_epsilon = 1e-3
    mean_fsc[mean_fsc < fsc_epsilon] = fsc_epsilon
    mean_fsc[mean_fsc > 1 - fsc_epsilon] = 1 - fsc_epsilon

    logger.debug(f"Halfset FSC: {mean_fsc[:L//2].cpu().numpy()}")

    fourier_reg = 1 / ((mean_fsc / (1 - mean_fsc)) * averaged_filter_gain)

    fourier_reg = upsample_and_expand_fourier_shell(fourier_reg.unsqueeze(0), rhs1.shape[-1], 3)

    reg_lhs = regularize_lhs(lhs1 + lhs2 + fourier_reg)
    mean_volume = (rhs1 + rhs2) / reg_lhs
    mean_volume = centered_ifft3(mean_volume, cropping_size=(L,) * 3).real

    if upsampling_factor > 1:
        mean_volume = correct_upsampling_padding(reg_lhs, rhs1 + rhs2, mean_volume, L, upsampling_factor)

    mean_volume = centered_ifft3(
        centered_fft3(mean_volume)
        * torch.tensor(grid_3d(L, shifted=False, normalized=True)["r"] <= 1, device=mean_volume.device)
    ).real

    if do_grid_correction:
        grid_correction = get_grid_correction(L, upsampling_factor, "nearest").to(mean_volume.device)
        mean_volume = mean_volume / grid_correction

    if mask is not None:
        mean_volume *= mask.squeeze(0)

    return mean_volume


def reconstruct_from_source(source: ImageSource, noise_var: float, lazy: bool = False) -> torch.Tensor:
    """Reconstruct the mean volume from an image source.

    Args:
        source (ImageSource): Raw data to be reconstructed from.
        noise_var (float): The estimated noise variance of the images.
        lazy (bool, optional): Whether to use a lazy dataset (default: False).

    Returns:
        torch.Tensor: The reconstructed mean volume.
    """
    dataset_cls = CovarDataset if not lazy else LazyCovarDataset

    mean_dataset = dataset_cls(source, noise_var, apply_preprocessing=False)
    mean_dataset.to_fourier_domain()

    reconstructed_mean = reconstruct_mean_from_halfsets(mean_dataset)

    torch.cuda.empty_cache()
    return reconstructed_mean


if __name__ == "__main__":
    import os
    import pickle

    from aspire.utils import Rotation
    from aspire.volume import Volume

    # dataset_path = 'data/pose_opt_exp'
    dataset_path = "data/scratch_data/igg_1d/images/snr0.01/downsample_L128/abinit_refine"
    dataset = pickle.load(open(os.path.join(dataset_path, "result_data/dataset.pkl"), "rb"))
    dataset.to_fourier_domain()

    USE_GT = True

    if USE_GT:
        gt_data = pickle.load(open(os.path.join(dataset_path, "result_data/gt_data.pkl"), "rb"))
        vol = gt_data.mean.unsqueeze(0).to("cuda:0")
        dataset.pts_rot = dataset.compute_pts_rot(
            torch.tensor(Rotation(gt_data.rotations.numpy()).as_rotvec()).to(torch.float32)
        )
    else:
        vol = torch.tensor(Volume.load("data/pose_opt_exp/relion_noisy_pose.mrc").asnumpy()).to("cuda:0")

    mask = (
        torch.tensor(Volume.load("data/scratch_data/igg_1d/init_mask/mask.mrc").downsample(vol.shape[-1]).asnumpy())
        .to("cuda:0")
        .squeeze(0)
    )
    rec_vol = reconstruct_mean_from_halfsets(dataset, batch_size=2048, mask=mask)

    fsc = vol_fsc(vol.squeeze(0), rec_vol.squeeze(0))
    mean_fsc = fsc[: vol.shape[-1] // 2].mean()
    mse_error = torch.norm(vol - rec_vol).cpu().numpy() / torch.norm(vol).cpu().numpy()

    print(f"Mean FSC : {mean_fsc}, MSE : {mse_error}")

    print(f"GT NORM : {torch.norm(vol)} ,EST NORM : {torch.norm(rec_vol)}")

    vols_rpsd = rpsd(*torch.concat((vol, rec_vol.unsqueeze(0)), dim=0)).cpu()

    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(fsc.cpu())
    axs[0, 1].plot(vols_rpsd.T.cpu()[:32])
    axs[0, 1].set_yscale("log")
    axs[1, 0].plot(vols_rpsd[0, :32] / vols_rpsd[1, :32])
    axs[1, 0].set_yscale("log")

    fig.savefig("reconstruct.jpg")

    Volume(vol.cpu().numpy()).save("exp_data/mean1.mrc", overwrite=True)
    Volume(rec_vol.cpu().numpy()).save("exp_data/mean2.mrc", overwrite=True)
