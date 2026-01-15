import logging
import os
import pickle
from typing import Any, Optional

import numpy as np
import torch
from aspire.volume import Volume

from solvar.utils import get_mpi_cpu_count, sub_starfile
from solvar.wiener_coords import mahalanobis_distance, mahalanobis_threshold

logger = logging.getLogger(__name__)


def relionReconstruct(
    inputfile: str,
    outputfile: str,
    classnum: Optional[int] = None,
    overwrite: bool = True,
    mrcs_index: Optional[Any] = None,
    invert: bool = False,
) -> Any:
    """Reconstruct volume using RELION.

    Args:
        inputfile: Input star file path
        outputfile: Output volume file path
        classnum: Class number to reconstruct (will use only the particles with the given class if exists) (optional)
        overwrite: Whether to overwrite existing output (default: True)
        mrcs_index: Indices for subset (optional)
        invert: Whether to invert volume (default: False)

    Returns:
        Reconstructed volume
    """
    if mrcs_index is not None:
        subfile = f"{inputfile}.sub.tmp"
        sub_starfile(inputfile, subfile, mrcs_index)
        inputfile = subfile
    classnum_arg = f" --class {classnum}" if classnum is not None else ""
    inputfile_path, inputfile_name = os.path.split(inputfile)
    outputfile_abs = os.path.abspath(outputfile)
    if overwrite or (not os.path.isfile(outputfile)):
        relion_command = f"relion_reconstruct --i {inputfile_name} --o {outputfile_abs} --ctf" + classnum_arg
        num_cores = get_mpi_cpu_count()
        if num_cores > 1:
            relion_command = (
                f'mpirun -np {num_cores} {relion_command.replace("relion_reconstruct","relion_reconstruct_mpi")}'
            )
        os.system(f"cd {inputfile_path} && {relion_command}")
        # compensate for volume sign inversion and normalization by image size in relion
        vol = -1 * Volume.load(outputfile)
        vol *= vol.shape[-1]
        if invert:
            vol = -1 * vol
        vol.save(outputfile, overwrite=True)
    else:
        vol = Volume.load(outputfile)
    if mrcs_index is not None:
        os.remove(subfile)
    return vol


def relionReconstructFromEmbedding(
    inputfile: str, outputfolder: str, embedding_positions: np.ndarray, q: float = 0.95
) -> None:
    """Reconstruct volumes from embedding positions using RELION.

    Args:
        inputfile: Input pickle file path
        outputfolder: Output folder path
        embedding_positions: Embedding positions array
        q: Quantile for outlier removal (default: 0.95)
    """
    with open(inputfile, "rb") as f:
        result = pickle.load(f)
    zs = torch.tensor(result["coords_est"])
    cov_zs = torch.tensor(result["coords_covar_inv_est"])
    starfile = result["particles_path"]
    assert isinstance(starfile, str) and starfile.endswith(
        ".star"
    ), f"Invalid particles file {starfile} reconstruction with relion only support starfiles"
    volumes_dir = os.path.join(outputfolder, "all_volumes_relion")
    os.makedirs(volumes_dir, exist_ok=True)
    for i, embedding_position in enumerate(torch.tensor(embedding_positions)):
        # Find closest point in zs
        embed_pos_index = torch.argmin(torch.norm(embedding_position - zs, dim=1))
        # Compute closest neighbors
        index_under_threshold = mahalanobis_threshold(zs, zs[embed_pos_index], cov_zs[embed_pos_index], q=q)
        logger.info(f"Reconstructing state {i} with {torch.sum(index_under_threshold)} images")
        output_file = os.path.join(volumes_dir, f"volume{i:04}.mrc")
        relionReconstruct(
            starfile,
            output_file,
            overwrite=True,
            mrcs_index=index_under_threshold.cpu().numpy(),
            invert=result["data_sign_inverted"],
        )


def reprojectVolumeFromEmbedding(inputfile, outputfolder, embedding_positions):
    with open(inputfile, "rb") as f:
        data = pickle.load(f)
    volumes_dir = os.path.join(outputfolder, "reprojected_volumes")
    os.makedirs(volumes_dir, exist_ok=True)
    eigenvecs = data["eigen_est"]
    mean_volume = data["mean_est"]
    reprojected_volumes = np.tensordot(embedding_positions, eigenvecs, axes=([1], [0])) + mean_volume

    for i, vol in enumerate(reprojected_volumes):
        output_file = os.path.join(volumes_dir, f"volume{i:04}.mrc")
        Volume(vol).save(output_file, overwrite=True)


def relionReconstructFromEmbeddingDisjointSets(inputfile, outputfolder, embedding_positions):
    with open(inputfile, "rb") as f:
        result = pickle.load(f)
    zs = torch.tensor(result["coords_est"])
    cov_zs = torch.tensor(result["coords_covar_inv_est"])
    starfile = result["particles_path"]
    assert isinstance(starfile, str) and starfile.endswith(
        ".star"
    ), f"Invalid particles file {starfile} reconstruction with relion only support starfiles"
    volumes_dir = os.path.join(outputfolder, "all_volumes_relion")
    os.makedirs(volumes_dir, exist_ok=True)
    mahal_distance = torch.zeros(zs.shape[0], embedding_positions.shape[0])
    for i, embedding_position in enumerate(torch.tensor(embedding_positions)):
        # Find closest point in zs
        embed_pos_index = torch.argmin(torch.norm(embedding_position - zs, dim=1))
        # Compute closest neighbors
        mahal_distance[:, i] = mahalanobis_distance(zs, zs[embed_pos_index], cov_zs[embed_pos_index])

    closest_embedding = mahal_distance.argmin(dim=1)
    for i, embedding_position in enumerate(torch.tensor(embedding_positions)):
        image_idx = closest_embedding == i
        logger.info(f"Reconstructing state {i} with {torch.sum(image_idx)} images")
        output_file = os.path.join(volumes_dir, f"volume{i:04}.mrc")
        relionReconstruct(
            starfile,
            output_file,
            overwrite=True,
            mrcs_index=image_idx.cpu().numpy(),
            invert=result["data_sign_inverted"],
        )
