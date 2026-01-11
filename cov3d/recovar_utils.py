import logging
import os
import pickle
from typing import Any, Optional, Tuple, Union

import numpy as np
import recovar
import torch
from cryodrgn.source import ImageSource
from recovar import dataset as recovar_ds
from recovar import output as recovar_output
from scipy.ndimage import binary_dilation

from cov3d.poses import pad_poses_by_ind

logger = logging.getLogger(__name__)


def getRecovarDataset(
    particles_path: str,
    ctf_path: Optional[str] = None,
    poses_path: Optional[str] = None,
    ind: Optional[np.ndarray] = None,
    split: bool = True,
    perm: Optional[np.ndarray] = None,
    uninvert_data: bool = False,
    lazy: bool = False,
) -> Tuple[Any, Optional[np.ndarray]]:
    """Get RECOVAR dataset from file paths.

    Args:
        particles_path: Path to particles file
        ctf_path: Path to CTF file (optional)
        poses_path: Path to poses file (optional)
        split: Whether to split dataset into two halves
        perm: Permutation array for splitting (optional)
        uninvert_data: Whether to uninvert data

    Returns:
        Tuple of (dataset, permutation_array)
    """

    particles_dir, _ = os.path.split(particles_path)
    dataset_dict = {"datadir": None, "uninvert_data": uninvert_data}
    dataset_dict["ctf_file"] = os.path.join(particles_dir, "ctf.pkl") if ctf_path is None else ctf_path
    dataset_dict["poses_file"] = os.path.join(particles_dir, "poses.pkl") if poses_path is None else poses_path
    dataset_dict["particles_file"] = particles_path

    if split:
        num_ims = ImageSource.from_file(dataset_dict["particles_file"]).n if ind is None else len(ind)
        if perm is None:
            perm = np.random.permutation(num_ims) if ind is None else np.random.permutation(ind)
        ind_split = [perm[: num_ims // 2], perm[num_ims // 2 :]]

        # Re-order halfsets for faster reading speed
        ind_split = [np.sort(ind_split[0]), np.sort(ind_split[1])]
        perm = np.concatenate(ind_split)

        if ind is not None:
            # Get perm indices with respect to ind
            # (such that ind[perm] will give the original perm)
            _, perm = np.unique(perm, return_inverse=True)

        return recovar_ds.get_split_datasets_from_dict(dataset_dict, ind_split, lazy=lazy), perm
    else:
        dataset_dict["ind"] = ind
        return recovar_ds.load_dataset_from_dict(dataset_dict, lazy=lazy), None


def recovarReconstruct(inputfile: str, outputfile: str, overwrite: bool = True, compute_mask: bool = False) -> None:
    """Reconstruct volume using RECOVAR.

    Args:
        inputfile: Path to input particles file
        outputfile: Path to output volume file
        overwrite: Whether to overwrite existing output file
        compute_mask: Whether to compute and save volume mask
    """
    if overwrite or (not os.path.isfile(outputfile)):
        dataset, _ = getRecovarDataset(inputfile)
        batch_size = recovar.utils.get_image_batch_size(
            dataset[0].grid_size, gpu_memory=recovar.utils.get_gpu_memory_total()
        )
        noise_variance, _ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)
        mean = recovar.homogeneous.get_mean_conformation_relion(
            dataset, batch_size=batch_size, noise_variance=noise_variance, use_regularization=True
        )
        recovar_output.save_volume(mean[0]["combined"], outputfile.replace(".mrc", ""), from_ft=True)

        if compute_mask:
            volume_mask = recovar.mask.make_mask_from_half_maps_from_means_dict(mean[0], smax=3)
            kernel_size = 3
            dilation_iterations = np.ceil(6 * dataset[0].volume_shape[0] / 128).astype(int)
            dilated_volume_mask = binary_dilation(volume_mask, iterations=dilation_iterations)
            volume_mask = recovar.mask.soften_volume_mask(volume_mask, kernel_size)
            dilated_volume_mask = recovar.mask.soften_volume_mask(dilated_volume_mask, kernel_size)
            recovar_output.save_volume(dilated_volume_mask, outputfile.replace(".mrc", "_mask"), from_ft=False)


def torch_to_numpy(arr: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensor to numpy array if needed.

    Args:
        arr: Input tensor or array

    Returns:
        Numpy array
    """
    return arr.numpy() if isinstance(arr, torch.Tensor) else arr


def check_dataset_size_limit(particles_path: str, ind: Optional[np.ndarray] = None) -> bool:
    """Check if the dataset size corresponding to the given particles file and indices is within a
    set limit.

    Args:
        particles_path: Path to the particles file.
        ind: Optional array of indices specifying which particles to use. If None, all particles are considered.

    Returns:
        True if the estimated dataset size is less than the maximum allowed limit (10 GB), False otherwise.
    """
    MAX_DATASET_LIMIT = 100 * (2**30)  # 100 GB
    image_source = ImageSource.from_file(particles_path, indices=ind)

    num_ims = image_source.n
    im_size = image_source.D
    dtype = np.dtype(image_source.dtype)
    dtype_size = dtype.itemsize

    dataset_size = num_ims * (im_size**2) * dtype_size

    return dataset_size < MAX_DATASET_LIMIT


def prepareDatasetForReconstruction(result_path: str) -> Tuple[Any, np.ndarray, np.ndarray, float, np.ndarray]:
    """Prepare dataset for reconstruction from result file.

    Args:
        result_path: Path to result pickle file

    Returns:
        Tuple of (dataset, coordinates, covariance_inverse, noise_variance, permutation)
    """
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    particles_path = result["particles_path"]
    ctf_path = result.get("ctf_path", None)
    poses_path = result.get("poses_path", None)
    ind = result.get("ind", None)

    if ind is not None:
        with open(poses_path, "rb") as f:
            poses = pickle.load(f)
        # Check if ind referes only to a subset of particles, and poses also refer to a subset
        if len(ind) == len(poses[0]) and len(ind) != (torch.max(ind) + 1):
            logger.warning(
                (
                    "Provided poses only match the indexed provided --ind argument, "
                    "but a full poses of all particles is expected, rewriting poses to a new file"
                )
            )
            padded_poses = pad_poses_by_ind(poses, ind)
            new_poses_path = poses_path + "_padded"
            with open(new_poses_path, "wb") as f:
                pickle.dump(padded_poses, f)
            poses_path = new_poses_path

    # TODO: Force lazy if dataset is too big - current RECOVAR's lazy implentation seem to be slow?
    lazy = False
    dataset, dataset_perm = getRecovarDataset(
        particles_path,
        ctf_path=ctf_path,
        poses_path=poses_path,
        ind=ind,
        uninvert_data=result["data_sign_inverted"],
        lazy=lazy,
    )
    batch_size = recovar.utils.get_image_batch_size(
        dataset[0].grid_size, gpu_memory=recovar.utils.get_gpu_memory_total()
    )
    noise_variance, _ = recovar.noise.estimate_noise_variance(dataset[0], batch_size)

    zs = result["coords_est"][dataset_perm]
    cov_zs = result["coords_covar_inv_est"][dataset_perm]

    return dataset, zs, cov_zs, noise_variance, dataset_perm


def recovarReconstructFromEmbedding(
    inputfile: str, outputfolder: str, embedding_positions: Union[str, np.ndarray], n_bins: int = 30
) -> None:
    """Reconstruct volumes from embedding positions using RECOVAR.

    Args:
        inputfile: Path to input result file
        outputfolder: Path to output folder
        embedding_positions: Embedding positions (file path or array)
        n_bins: Number of bins for reconstruction
    """
    dataset, zs, cov_zs, noise_variance, _ = prepareDatasetForReconstruction(inputfile)
    L = dataset[0].grid_size
    B_factor = 0  # TODO: handle B_factor
    if os.path.isfile(embedding_positions):
        with open(embedding_positions, "rb") as f:
            embedding_positions = pickle.load(f)

    recovar_output.compute_and_save_reweighted(
        dataset,
        embedding_positions,
        zs,
        cov_zs,
        noise_variance * np.ones(L // 2 - 1),
        outputfolder,
        B_factor,
        n_bins=n_bins,
    )
