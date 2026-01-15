import logging
import multiprocessing
import os
import re
from typing import Any, List, Optional, Tuple, Union

import aspire
import numpy as np
import pandas as pd
import starfile
import torch
from aspire.storage.starfile import StarFile
from aspire.utils import coor_trans, grid_2d, grid_3d
from aspire.volume import Volume
from numpy import random
from sklearn.decomposition import PCA

from solvar.projection_funcs import centered_fft2, centered_fft3

logger = logging.getLogger(__name__)


def generateBallVoxel(center: Tuple[float, float, float], radius: float, L: int) -> np.ndarray:
    """Generate a ball-shaped voxel.

    Args:
        center: Center coordinates (x, y, z)
        radius: Radius of the ball
        L: Volume resolution

    Returns:
        Ball voxel as flattened array of shape (1, L^3)
    """
    grid = coor_trans.grid_3d(L)
    voxel = ((grid["x"] - center[0]) ** 2 + (grid["y"] - center[1]) ** 2 + (grid["z"] - center[2]) ** 2) <= np.power(
        radius, 2
    )

    return np.single(voxel.reshape((1, L**3)))


def generateCylinderVoxel(center: Tuple[float, float], radius: float, L: int, axis: int = 2) -> np.ndarray:
    """Generate a cylinder-shaped voxel.

    Args:
        center: Center coordinates in the cylinder plane
        radius: Radius of the cylinder
        L: Volume resolution
        axis: Axis along which the cylinder extends (0=x, 1=y, 2=z)

    Returns:
        Cylinder voxel as flattened array of shape (1, L^3)
    """
    grid = coor_trans.grid_3d(L)
    dims = ("x", "y", "z")
    cylinder_axes = tuple(dims[i] for i in range(3) if i != axis)
    voxel = ((grid[cylinder_axes[0]] - center[0]) ** 2 + (grid[cylinder_axes[1]] - center[1]) ** 2) <= np.power(
        radius, 2
    )

    return np.single(voxel.reshape((1, L**3)))


def replicateVoxelSign(voxels: Volume) -> Volume:
    """Replicate volumes with sign flipped versions.

    Args:
        voxels: Input volumes

    Returns:
        Volume with both original and sign-flipped versions
    """
    return Volume(np.concatenate((voxels.asnumpy(), -voxels.asnumpy()), axis=0))


def volsCovarEigenvec(
    vols: Union[Volume, np.ndarray],
    eigenval_threshold: float = 1e-3,
    randomized_alg: bool = False,
    max_eigennum: Optional[int] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute covariance eigenvectors from volumes.

    Args:
        vols: Input volumes
        eigenval_threshold: Threshold for eigenvalue selection
        randomized_alg: Whether to use randomized SVD algorithm
        max_eigennum: Maximum number of eigenvectors (for randomized algorithm)
        weights: Optional weights for volumes

    Returns:
        Eigenvectors scaled by square root of eigenvalues
    """
    vols_num = vols.shape[0]
    if weights is None:  # If
        vols_dist = np.ones(vols_num) / vols_num
    else:
        vols_dist = weights / np.sum(weights)
    vols_dist = vols_dist.astype(vols.dtype)
    vols_mean = np.sum(vols_dist[:, np.newaxis, np.newaxis, np.newaxis] * vols, axis=0)
    vols0mean = asnumpy((vols - vols_mean)).reshape((vols_num, -1))

    if not randomized_alg:
        vols0mean = np.sqrt(vols_dist[:, np.newaxis]) * vols0mean
        _, volsSTD, volsSpan = np.linalg.svd(vols0mean, full_matrices=False)
        # volsSTD /= np.sqrt(vols_num)  #standard devation is volsSTD / sqrt(n)
        eigenval_num = np.sum(volsSTD > np.sqrt(eigenval_threshold))
        volsSpan = volsSpan[:eigenval_num, :] * volsSTD[:eigenval_num, np.newaxis]
    else:
        # TODO : add weights to randomized alg
        if max_eigennum is None:
            max_eigennum = vols_num
        pca = PCA(n_components=max_eigennum, svd_solver="randomized")
        fitvols = pca.fit(vols0mean)
        volsSpan = fitvols.components_ * np.sqrt(fitvols.explained_variance_.reshape((-1, 1)))

    return volsSpan


def rademacherDist(sz):
    val = random.randint(0, 2, sz)
    val[val == 0] = -1
    return val


def nonNormalizedGS(vecs):
    vecnum = vecs.shape[0]
    ortho_vecs = torch.zeros(vecs.shape)
    ortho_vecs[0] = vecs[0]
    for i in range(1, vecnum):
        ortho_vecs[i] = vecs[i]
        for j in range(i):
            ortho_vecs[i] = (
                ortho_vecs[i] - torch.sum(vecs[i] * ortho_vecs[j]) / (torch.norm(ortho_vecs[j]) ** 2) * ortho_vecs[j]
            )

    return ortho_vecs


def cosineSimilarity(vec1, vec2):

    vec1 = vec1.reshape((vec1.shape[0], -1))
    vec2 = vec2.reshape((vec2.shape[0], -1))
    vec1 = torch.linalg.svd(vec1, full_matrices=False)[2]
    vec2 = torch.linalg.svd(vec2, full_matrices=False)[2]
    cosine_sim = torch.matmul(vec1, torch.transpose(vec2, 0, 1).conj()).cpu().numpy()

    return cosine_sim


def principalAngles(vec1, vec2):

    vec1 = asnumpy(vec1).reshape((vec1.shape[0], -1))
    vec2 = asnumpy(vec2).reshape((vec2.shape[0], -1))

    svd1 = np.linalg.svd(vec1, full_matrices=False)[2]
    svd2 = np.linalg.svd(vec2, full_matrices=False)[2]

    principal_angles = np.arccos(np.clip(np.abs(np.dot(svd1, svd2.T)), -1.0, 1.0))

    return np.min(np.degrees(principal_angles))


def frobeniusNorm(vecs):
    # returns the frobenius norm of a matrix given by its eigenvectors (multiplied by the corresponding sqrt(eigenval))
    vecs = asnumpy(vecs).reshape((vecs.shape[0], -1))
    vecs_inn_prod = np.matmul(vecs, vecs.transpose())
    return np.sqrt(np.sum(vecs_inn_prod**2))


def frobeniusNormDiff(vec1, vec2):
    # returns the frobenius norm of the diffrence of two matrices given by their
    # eigenvectors (multiplied by the corresponding sqrt(eigenval))

    vec1 = asnumpy(vec1).reshape((vec1.shape[0], -1))
    vec2 = asnumpy(vec2).reshape((vec2.shape[0], -1))

    normdiff_squared = (
        frobeniusNorm(vec1) ** 2 + frobeniusNorm(vec2) ** 2 - 2 * np.sum(np.matmul(vec1, vec2.transpose()) ** 2)
    )

    return np.sqrt(normdiff_squared)


def randomized_svd(A_mv, dim, rank, num_iters=10):
    # Generate random test matrix
    Q = torch.randn(dim, rank)
    for _ in range(num_iters):
        # Orthogonalize
        Q, _ = torch.linalg.qr(A_mv(Q))
        Q = Q[:, :rank]  # Reduce to desired rank

    # Compute B = Q^T A
    B = torch.zeros(rank, dim)
    for i in range(rank):
        B[i] = A_mv(Q[:, i])

    # Compute SVD of the smaller matrix B
    U_tilde, S, V = torch.svd(B)

    # Project U_tilde back to the original space
    U = Q @ U_tilde
    return U, S, V


def asnumpy(data):
    if isinstance(data, (aspire.volume.volume.Volume, aspire.image.image.Image)):
        return data.asnumpy()

    return data


def np2torchDtype(np_dtype):

    return torch.float64 if (np_dtype == np.double) else torch.float32


dtype_mapping = {
    torch.float16: torch.complex32,
    torch.complex32: torch.float16,
    torch.float32: torch.complex64,
    torch.complex64: torch.float32,
    torch.float64: torch.complex128,
    torch.complex128: torch.float64,
}


def get_complex_real_dtype(dtype):
    return dtype_mapping[dtype]


def get_torch_device() -> torch.device:
    """Get the current PyTorch device (GPU if available, otherwise CPU).

    Returns:
        PyTorch device object
    """
    return torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")


def appendCSV(dataframe: Any, csv_file: str) -> None:
    """Append a dataframe to a CSV file.

    Args:
        dataframe: DataFrame to append
        csv_file: Path to CSV file
    """
    if os.path.isfile(csv_file):
        current_dataframe = pd.read_csv(csv_file, index_col=0)
        updated_dataframe = pd.concat([current_dataframe, dataframe], ignore_index=True)
        updated_dataframe.to_csv(csv_file)

    else:
        dataframe.to_csv(csv_file)


def soft_edged_kernel(
    radius: float, L: int, dim: int, radius_backoff: int = 2, in_fourier: bool = False
) -> torch.Tensor:
    """Create a soft-edged kernel in real or Fourier space.

    Implementation based on RECOVAR: https://github.com/ma-gilles/recovar/blob/main/recovar/mask.py#L106

    Args:
        radius: Radius of the kernel
        L: Box size (assumed square/cubic)
        dim: Dimension (2 or 3)
        radius_backoff: Soft edge width (default: 2)
        in_fourier: Whether to create kernel in Fourier space (default: False)

    Returns:
        Soft-edged kernel tensor
    """
    # Implementation is based on RECOVAR https://github.com/ma-gilles/recovar/blob/main/recovar/mask.py#L106
    if radius < 3:
        radius = 3
        logger.warning(f"Radius value {radius} is too small. setting radius to 3 pixels.")
    if dim == 2:
        grid_func = grid_2d
    elif dim == 3:
        grid_func = grid_3d

    grid_radius = grid_func(L, shifted=True, normalized=False)["r"]
    radius0 = radius - radius_backoff

    kernel = np.zeros(grid_radius.shape)

    kernel[grid_radius < radius0] = 1

    kernel = np.where(
        (grid_radius >= radius0) * (grid_radius < radius),
        (1 + np.cos(np.pi * (grid_radius - radius0) / (radius - radius0))) / 2,
        kernel,
    )

    kernel = torch.tensor(kernel / np.sum(kernel))
    if in_fourier:
        kernel = centered_fft2(kernel) if dim == 2 else centered_fft3(kernel)
    return kernel


def project_mean_out_from_eigenvecs(eigenvecs: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:

    orig_shape = eigenvecs.shape
    r = eigenvecs.shape[0]
    eigenvecs = eigenvecs.reshape(r, -1)
    mean = mean.reshape(1, -1)

    inn_prod = torch.matmul(eigenvecs, mean.conj().T)
    mean_norm_squared = torch.norm(mean) ** 2

    projected_eigenvecs = eigenvecs - torch.matmul(inn_prod, mean) / mean_norm_squared

    _, S, V = torch.linalg.svd(projected_eigenvecs, full_matrices=False)

    return (S.reshape(-1, 1) * V).reshape(orig_shape)


def meanCTFPSD(ctfs: List[Any], L: int) -> np.ndarray:
    """Compute mean CTF power spectral density.

    Args:
        ctfs: List of CTF objects
        L: Grid size

    Returns:
        Mean PSD array
    """
    ctfs_eval_grid = [np.power(ctf.evaluate_grid(L), 2) for ctf in ctfs]
    return np.mean(np.array(ctfs_eval_grid), axis=0)


def sub_starfile(star_input: str, star_output: str, mrcs_index: Any) -> None:
    """Create a subset of a star file based on given indices.

    Args:
        star_input: Input star file path
        star_output: Output star file path
        mrcs_index: Indices to subset
    """

    # Read the star file as a pandas DataFrame
    data = starfile.read(star_input)
    # If the input is a dict, we assume "particles" key; otherwise it is just the DataFrame
    if isinstance(data, dict) and "particles" in data:
        # Subset the particles dataframe and store as dict
        data["particles"] = data["particles"].iloc[mrcs_index]
    else:
        # Single DataFrame case
        data = data.iloc[mrcs_index]

    input_dir = os.path.dirname(os.path.abspath(star_input))
    output_dir = os.path.dirname(os.path.abspath(star_output))
    is_different_dir = input_dir != output_dir

    if is_different_dir:
        # Get relative path of output file's directory to input file's directory
        rel_output_dir = os.path.relpath(input_dir, output_dir)
        # Append rel_output_dir to _rlnImageName in data['particles']
        if isinstance(data, dict) and "particles" in data:
            particles = data["particles"]
            if "rlnImageName" in particles.columns:

                def append_rel_dir(im_name):
                    # Only prepend rel_output_dir if not already present and rel_output_dir is not '.'
                    if rel_output_dir != "." and not str(im_name).startswith(rel_output_dir + os.sep):
                        # The im_name is often like "1@oldname.mrcs" -> keep the prefix (e.g., "1@")
                        parts = str(im_name).split("@", 1)
                        if len(parts) == 2:
                            return f"{parts[0]}@{os.path.join(rel_output_dir, parts[1])}"
                        else:
                            return os.path.join(rel_output_dir, str(im_name))
                    else:
                        return im_name

                particles["rlnImageName"] = particles["rlnImageName"].apply(append_rel_dir)

    starfile.write(data, star_output)


def mrcs_replace_starfile(star_input: str, star_output: str, mrcs_name: str) -> None:
    """Replace MRCS file name in star file.

    Args:
        star_input: Input star file path
        star_output: Output star file path
        mrcs_name: New MRCS file name
    """
    star_out = StarFile(star_input)
    star_out["particles"]["_rlnImageName"] = [
        re.sub(r"@[^@]+", f"@{mrcs_name}", s) for s in star_out["particles"]["_rlnImageName"]
    ]
    star_out.write(star_output)


def vol_fsc(vol1: Union[torch.Tensor, Any], vol2: Union[torch.Tensor, Any]) -> Any:
    """Compute Fourier Shell Correlation between two volumes.

    Args:
        vol1: First volume
        vol2: Second volume

    Returns:
        FSC values

    Raises:
        Exception: If volumes are not of the same type
    """
    if not isinstance(vol2, type(vol1)):
        raise Exception(
            f"Volumes of the same type expected vol1 is of type {type(vol1)} while vol2 is of type {type(vol2)}"
        )

    if isinstance(vol1, aspire.volume.volume.Volume):
        return vol1.fsc(vol2)

    elif isinstance(vol1, torch.Tensor):
        # TODO : implement faster FSC for torch tensors
        vol1 = Volume(vol1.cpu().numpy())
        vol2 = Volume(vol2.cpu().numpy())

        return vol1.fsc(vol2)


def get_cpu_count() -> int:
    """Get the number of available CPUs, accounting for job schedulers.

    Returns:
        Number of available CPUs
    """

    # Check for SLURM environment variable first
    # TODO : handle other job schedulers
    slurm_cpu_count = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpu_count is not None:
        return int(slurm_cpu_count)

    return multiprocessing.cpu_count()


def get_mpi_cpu_count() -> int:
    """Get the number of MPI tasks, accounting for job schedulers.

    Returns:
        Number of MPI tasks
    """

    # Check for SLURM environment variable first
    # TODO : handle other job schedulers
    if "SLURM_JOB_ID" in os.environ:
        slurm_cpu_count = os.getenv("SLURM_NTASKS", 1)
        return int(slurm_cpu_count)
    # if slurm_cpu_count is not None:
    #    return int(slurm_cpu_count)

    return multiprocessing.cpu_count()


def readVols(vols, in_list=True):
    volfiles = [os.path.join(vols, v) for v in os.listdir(vols) if ".mrc" in v] if isinstance(vols, str) else vols
    numvols = len(volfiles)
    vol_size = Volume.load(volfiles[0]).shape[-1]
    volumes = Volume(np.zeros((numvols, vol_size, vol_size, vol_size), dtype=np.float32))
    pixel_size = [0 for i in range(numvols)]

    for i, volfile in enumerate(volfiles):
        v = Volume.load(volfile)
        volumes[i] = v
        pixel_size[i] = v.pixel_size

    assert all(p == pixel_size[0] for p in pixel_size), "All pixel sizes must be the same"

    if not in_list:
        return Volume(np.concatenate(volumes), pixel_size=pixel_size[0])

    return volumes


def saveVol(vols: torch.tensor, path: str):
    Volume(vols.cpu().numpy()).save(path, overwrite=True)


def create_mask_from_vols(vols: Volume, threshold: float):
    """Creates a simple binary mask from volumes."""

    return Volume(vols.asnumpy().mean(axis=0) > threshold)


def set_module_grad(module: torch.nn.Module, grad: bool):
    for param in module.parameters():
        param.requires_grad = grad
