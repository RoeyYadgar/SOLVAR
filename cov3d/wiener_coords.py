import math
from typing import Optional, Tuple, Union

import torch
from scipy.stats import chi2
from tqdm import tqdm

from cov3d.dataset import CovarDataset
from cov3d.nufft_plan import NufftPlan, NufftSpec
from cov3d.projection_funcs import make_nufft_plan, vol_forward


def wiener_coords(
    dataset: CovarDataset,
    eigenvecs: torch.Tensor,
    eigenvals: torch.Tensor,
    batch_size: int = 1024,
    start_ind: Optional[int] = None,
    end_ind: Optional[int] = None,
    return_eigen_forward: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute Wiener-filtered latent coordinates from eigenvectors.

    Args:
        dataset: Dataset containing particle images and poses
        eigenvecs: Eigenvectors of shape (rank, L, L, L)
        eigenvals: Eigenvalues of shape (rank,) or diagonal matrix of shape (rank, rank)
        batch_size: Batch size for processing
        start_ind: Starting index for processing (default: 0)
        end_ind: Ending index for processing (default: len(dataset))
        return_eigen_forward: Whether to return eigenvector projections

    Returns:
        Latent coordinates or tuple of (coordinates, eigen_forward_images)
    """
    if start_ind is None:
        start_ind = 0
    if end_ind is None:
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device

    covar_noise = dataset.noise_var * torch.eye(rank, device=device)
    if len(eigenvals.shape) == 1:
        eigenvals = torch.diag(eigenvals)

    nufft_plans = NufftPlan((L,) * 3, batch_size=rank, dtype=dtype, device=device)
    coords = torch.zeros((end_ind - start_ind, rank), device=device)
    if return_eigen_forward:
        eigen_forward_images = torch.zeros((end_ind - start_ind, rank, L, L), dtype=dtype)

    pbar = tqdm(total=math.ceil(coords.shape[0] / batch_size), desc="Computing latent coordinates")
    for i in range(0, coords.shape[0], batch_size):
        images, pts_rot, batch_filters, _ = dataset[start_ind + i : min(start_ind + i + batch_size, end_ind)]
        num_ims = images.shape[0]
        pts_rot = pts_rot.to(device)
        images = images.to(device).reshape(num_ims, -1)
        batch_filters = batch_filters.to(device)
        nufft_plans.setpts(pts_rot)

        eigen_forward = vol_forward(eigenvecs, nufft_plans, batch_filters)
        if return_eigen_forward:
            eigen_forward_images[i : i + num_ims] = eigen_forward.to("cpu")
        eigen_forward = eigen_forward.reshape((num_ims, rank, -1))

        for j in range(num_ims):
            eigen_forward_Q, eigen_forward_R = torch.linalg.qr(eigen_forward[j].T)
            image_coor = images[j] @ eigen_forward_Q
            image_coor_covar = eigen_forward_R @ eigenvals @ eigen_forward_R.T + covar_noise
            image_coor = eigenvals @ eigen_forward_R.T @ torch.inverse(image_coor_covar) @ image_coor
            coords[i + j, :] = image_coor

        pbar.update(1)
    pbar.close()

    if not return_eigen_forward:
        return coords
    else:
        return coords, eigen_forward_images


def latentMAP(
    dataset: CovarDataset,
    eigenvecs: torch.Tensor,
    eigenvals: torch.Tensor,
    batch_size: int = 1024,
    start_ind: Optional[int] = None,
    end_ind: Optional[int] = None,
    return_coords_covar: bool = False,
    nufft_spec: Optional[NufftSpec] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute maximum a posteriori latent coordinates.

    Args:
        dataset: Dataset containing particle images and poses
        eigenvecs: Eigenvectors of shape (rank, L, L, L)
        eigenvals: Eigenvalues of shape (rank,) or (rank, rank)
        batch_size: Batch size for processing
        start_ind: Starting index for processing (default: 0)
        end_ind: Ending index for processing (default: len(dataset))
        return_coords_covar: Whether to return coordinate covariance inverse
        nufft_plan: NUFFT plan class to use
        **nufft_plan_kwargs: Additional arguments for NUFFT plan

    Returns:
        Latent coordinates or tuple of (coordinates, coords_covar_inv)
    """
    if start_ind is None:
        start_ind = 0
    if end_ind is None:
        end_ind = len(dataset)
    vol_shape = eigenvecs.shape
    L = vol_shape[-1]
    rank = eigenvecs.shape[0]
    dtype = eigenvecs.dtype
    device = eigenvecs.device

    if len(eigenvals.shape) == 1:
        eigenvals = torch.diag(eigenvals)

    if nufft_spec is None:
        nufft_spec = NufftSpec(nufft_type=NufftPlan)
    nufft_spec.update(sz=(L,) * 3, batch_size=rank, dtype=dtype, device=device)
    nufft_plans, eigenvecs = make_nufft_plan(nufft_spec, eigenvecs)

    eigenvals_inv = torch.inverse(
        eigenvals + 1e-6 * torch.eye(rank, device=device, dtype=dtype)
    )  # add a small value to avoid numerical instability

    coords = torch.zeros((end_ind - start_ind, rank), device=device)
    if return_coords_covar:
        coords_covar_inv = torch.zeros((end_ind - start_ind, rank, rank), dtype=dtype)

    pbar = tqdm(total=math.ceil(coords.shape[0] / batch_size), desc="Computing latent coordinates")
    for i in range(0, coords.shape[0], batch_size):
        images, pts_rot, batch_filters, _ = dataset[start_ind + i : min(start_ind + i + batch_size, end_ind)]
        pts_rot = pts_rot.to(device)
        images = images.to(device)
        batch_filters = batch_filters.to(device)
        nufft_plans.setpts(pts_rot)

        eigen_forward = vol_forward(eigenvecs, nufft_plans, batch_filters)

        latent_coords, m, _ = compute_latentMAP_batch(images, eigen_forward, dataset.noise_var, eigenvals_inv)
        coords[i : (i + batch_size)] = latent_coords.squeeze(-1)

        if return_coords_covar:
            coords_covar_inv[i : (i + batch_size)] = m.to("cpu")

        pbar.update(1)
    pbar.close()

    del nufft_plans

    if not return_coords_covar:
        return coords
    else:
        return coords, coords_covar_inv


def compute_latentMAP_batch(
    images: torch.Tensor,
    eigen_forward: torch.Tensor,
    noise_var: float,
    eigenvals_inv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute MAP latent coordinates for a batch of images.

    Args:
        images: Input images of shape (batch, L, L)
        eigen_forward: Forward projections of eigenvectors of shape (batch, rank, L*L)
        noise_var: Noise variance
        eigenvals_inv: Inverse eigenvalues (default: identity matrix)

    Returns:
        Tuple of (latent_coords, covariance_matrix, projected_images)
    """
    n = images.shape[0]
    r = eigen_forward.shape[1]
    if eigenvals_inv is None:
        eigenvals_inv = torch.eye(r, device=eigen_forward.device, dtype=eigen_forward.dtype)
    images = images.reshape(n, -1, 1)
    eigen_forward = eigen_forward.reshape(n, r, -1)

    m = eigen_forward.conj() @ eigen_forward.transpose(1, 2) / noise_var + eigenvals_inv

    projected_images = torch.matmul(eigen_forward.conj(), images) / noise_var  # size (batch, rank,1)

    # There can be numerical instability with inverting the matrix m due to small entries
    # correct it here by normalizing the matrix by trace(m)/size(m) before inversion
    mean_m = m.diagonal(dim1=-2, dim2=-1).abs().sum(dim=1) / m.shape[-1]
    latent_coords = torch.linalg.solve(m / mean_m.reshape(-1, 1, 1), projected_images) / mean_m.reshape(
        -1, 1, 1
    )  # size (batch, rank,1)

    return latent_coords, m, projected_images


def mahalanobis_distance(
    coords: torch.Tensor, coords_mean: torch.Tensor, coords_covar_inv: torch.Tensor
) -> torch.Tensor:
    """Compute Mahalanobis distance for coordinates.

    Args:
        coords: Coordinate vectors of shape (n_samples, n_features)
        coords_mean: Mean coordinates of shape (n_features,)
        coords_covar_inv: Inverse covariance matrix of shape (n_features, n_features)

    Returns:
        Mahalanobis distances of shape (n_samples,)
    """
    mean_centered_coords = coords - coords_mean
    dist = torch.sum((mean_centered_coords @ (coords_covar_inv)) * mean_centered_coords, dim=1)

    return dist


def mahalanobis_threshold(
    coords: torch.Tensor, coords_mean: torch.Tensor, coords_covar_inv: torch.Tensor, q: float = 0.95
) -> torch.Tensor:
    """Compute Mahalanobis distance threshold for outlier detection.

    Args:
        coords: Coordinate vectors of shape (n_samples, n_features)
        coords_mean: Mean coordinates of shape (n_features,)
        coords_covar_inv: Inverse covariance matrix of shape (n_features, n_features)
        q: Quantile for threshold (default: 0.95)

    Returns:
        Boolean mask indicating inliers
    """
    dist = mahalanobis_distance(coords, coords_mean, coords_covar_inv)
    return dist < chi2.ppf(q, df=coords.shape[1])
