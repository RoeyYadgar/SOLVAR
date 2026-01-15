import logging
import os
import pickle

import numpy as np
import torch
from aspire.operators import ArrayFilter, RadialCTFFilter
from aspire.volume import LegacyVolume, Volume
from matplotlib import pyplot as plt

from solvar.analyze import analyze
from solvar.dataset import CovarDataset, GTData
from solvar.source import SimulatedSource
from solvar.utils import readVols, volsCovarEigenvec
from solvar.workflow import covar_processing, load_mask

logger = logging.getLogger(__name__)


def display_source(source: SimulatedSource, output_path: str, num_ims: int = 2, display_clean: bool = False) -> None:
    """Display sample images from the simulated source.

    Args:
        source: SimulatedSource instance
        output_path: Path to save the display image
        num_ims: Number of images to display per distinct volume
        display_clean: Whether to display clean or noisy images
    """
    num_vols = len(np.unique(source.states))
    fig, axs = plt.subplots(num_ims, num_vols, figsize=(2 * num_vols, 2 * num_ims))
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    im_samples = source._clean_images[:20].numpy() if display_clean else source.images[:20].asnumpy()
    im_min = im_samples.min()
    im_max = im_samples.max()

    for i in range(num_vols):
        state_inds = np.where(source.states == i)[0][:num_ims]
        clean_images = (
            source._clean_images[state_inds].numpy() if display_clean else source.images[state_inds].asnumpy()
        )
        for j in range(num_ims):
            axs_idx = (j, i) if num_vols > 1 else j
            axs[axs_idx].imshow(clean_images[j], cmap="gray", vmin=im_min, vmax=im_max)
            axs[axs_idx].set_xticks([])  # Remove x-axis ticks
            axs[axs_idx].set_yticks([])
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)


def replicate_source(source: SimulatedSource) -> SimulatedSource:
    """Replicate a simulated source by doubling all data.

    Args:
        source: SimulatedSource instance to replicate

    Returns:
        Modified source with doubled data
    """
    source.rotations = np.tile(source.rotations, (2, 1, 1))
    source.filter_indices = np.tile(source.filter_indices, (2))
    source.states = torch.tile(source.states, (2,))
    source.amplitudes = np.tile(source.amplitudes, (2))
    source.offsets = torch.tile(source.offsets, (2, 1))
    source._clean_images = torch.tile(source._clean_images, (2, 1, 1))
    source._image_noise = torch.tile(source._image_noise, (2, 1, 1))
    source.n = source.n * 2

    return source


def simulateExp(folder_name=None, L=64, r=5, no_ctf=False, save_source=False, vols=None, mask=None):
    os.makedirs(folder_name, exist_ok=True)

    n = 100000
    pixel_size = 3 * 128 / L

    if not no_ctf:
        filters = [
            RadialCTFFilter(defocus=d, pixel_size=pixel_size)
            for d in np.random.lognormal(np.log(20000), 0.3, size=(928))
        ]
    else:
        filters = [ArrayFilter(np.ones((L, L)))]

    if vols is None:
        voxels = LegacyVolume(L=int(L * 0.7), C=r + 1, K=64, dtype=np.float32, pixel_size=pixel_size).generate()
        padded_voxels = np.zeros((r + 1, L, L, L), dtype=np.float32)
        pad_width = (L - voxels.shape[1]) // 2
        padded_voxels[
            :,
            pad_width : pad_width + voxels.shape[1],
            pad_width : pad_width + voxels.shape[2],
            pad_width : pad_width + voxels.shape[3],
        ] = voxels
        voxels = Volume(padded_voxels)
        voxels.save(os.path.join(folder_name, "gt_vols.mrc"), overwrite=True)
    else:
        voxels = readVols(vols, in_list=False)

    sim = SimulatedSource(n, vols=voxels, unique_filters=filters, noise_var=0)
    var = torch.var(sim._clean_images).item()

    vectorsGT = volsCovarEigenvec(voxels)
    snr_vals = 10 ** np.arange(0, -3.5, -0.5)
    objs = ["ml", "ls"]
    for snr in snr_vals:
        noise_var = var / snr
        logger.info(f"Signal power : {var}. Using noise variance of {noise_var} to achieve SNR of {snr}")

        sim.noise_var = noise_var
        noise_var = sim.noise_var
        dataset = CovarDataset(
            sim,
            noise_var,
            mean_volume=Volume(voxels.asnumpy().mean(axis=0)),
            mask=Volume.load(mask) if mask is not None else None,
        )
        gt_data = GTData(vectorsGT)

        for obj in objs:
            dir_name = os.path.join(folder_name, f"obj_{obj}", f"algorithm_output_{snr}")
            os.makedirs(dir_name, exist_ok=True)
            if save_source:
                sim.save(dir_name)
            display_source(sim, os.path.join(dir_name, "clean_images.jpg"), display_clean=True)
            display_source(sim, os.path.join(dir_name, "noisy_images.jpg"), display_clean=False)
            data_dict, _, _ = covar_processing(
                dataset, r, dir_name, gt_data=gt_data, max_epochs=20, objective_func=obj, num_reg_update_iters=1
            )

            coords_est = data_dict["coords_est"]
            state_centers = np.zeros((len(voxels), coords_est.shape[1]))
            for i in range(len(voxels)):
                state_centers[i] = coords_est[sim.states == i].mean(axis=0)
            with open(os.path.join(dir_name, "state_centers.pkl"), "wb") as f:
                pickle.dump(state_centers, f)

            analyze(
                os.path.join(dir_name, "recorded_data.pkl"),
                output_dir=dir_name,
                analyze_with_gt=True,
                skip_reconstruction=True,
                gt_labels=sim.states,
                latent_coords=os.path.join(dir_name, "state_centers.pkl"),
            )


def simulate_noisy_rots(
    folder_name,
    snr=None,
    noise_var=None,
    rots_std=0,
    offsets_std=0,
    L=64,
    r=5,
    n=100000,
    no_ctf=False,
    vols=None,
    mask=None,
):
    os.makedirs(folder_name, exist_ok=True)

    if vols is None:
        pixel_size = 3 * 128 / L
        voxels = LegacyVolume(L=int(L * 0.7), C=r + 1, K=64, dtype=np.float32, pixel_size=pixel_size).generate()
        padded_voxels = np.zeros((r + 1, L, L, L), dtype=np.float32)
        pad_width = (L - voxels.shape[1]) // 2
        padded_voxels[
            :,
            pad_width : pad_width + voxels.shape[1],
            pad_width : pad_width + voxels.shape[2],
            pad_width : pad_width + voxels.shape[3],
        ] = voxels
        voxels = Volume(padded_voxels)
        voxels.save(os.path.join(folder_name, "gt_vols.mrc"), overwrite=True)
        pixel_size = 3 * 128 / L
    else:
        voxels = Volume.load(vols) if isinstance(vols, str) else readVols(vols, in_list=False)
        if voxels.resolution > L:
            voxels = voxels.downsample(L)
        pixel_size = voxels.pixel_size

    if not no_ctf:
        # filters = [RadialCTFFilter(defocus=d,pixel_size=pixel_size) for d in np.linspace(8e3, 2.5e4, 927)]
        filters = [
            RadialCTFFilter(defocus=d, pixel_size=pixel_size)
            for d in np.random.lognormal(np.log(20000), 0.3, size=(928))
        ]
    else:
        filters = [ArrayFilter(np.ones((L, L)))]

    sim = SimulatedSource(
        n, vols=voxels, unique_filters=filters, noise_var=0, rotations_std=rots_std, offsets_std=offsets_std
    )
    var = torch.var(sim._clean_images).item()

    assert (snr is None) + (noise_var is None) == 1
    if noise_var is None:
        noise_var = var / snr
    sim.noise_var = noise_var
    noise_var = sim.noise_var

    # Place mean est and class vols in the output dir
    output_dir = os.path.join(folder_name, "result_data")
    os.makedirs(output_dir, exist_ok=True)
    mean = voxels.asnumpy().mean(axis=0)
    Volume(mean, pixel_size=pixel_size).save(os.path.join(output_dir, "mean_est.mrc"), overwrite=True)
    voxels.save(os.path.join(output_dir, "class_vols.mrc"), overwrite=True)
    vectorsGT = volsCovarEigenvec(voxels)
    dataset = CovarDataset(sim, noise_var, mean_volume=mean, mask=load_mask(mask, L), apply_preprocessing=False)

    gt_data = GTData(vectorsGT, mean, sim._rotations, sim._offsets)

    sim.save(folder_name, gt_pose=False)
    os.makedirs(os.path.join(folder_name, "gt"), exist_ok=True)
    sim.save(os.path.join(folder_name, "gt"), save_image_stack=False, gt_pose=True)
    display_source(sim, os.path.join(folder_name, "clean_images.jpg"), display_clean=True)
    display_source(sim, os.path.join(folder_name, "noisy_images.jpg"), display_clean=False)
    with open(os.path.join(output_dir, "dataset.pkl"), "wb") as f:
        pickle.dump(dataset, f)
    with open(os.path.join(output_dir, "gt_data.pkl"), "wb") as f:
        pickle.dump(gt_data, f)


if __name__ == "__main__":
    ribo_vols = [
        os.path.join("data/scratch_data/cryodrgn_ribosomes/ribosomes/inputs", v)
        for v in os.listdir("data/scratch_data/cryodrgn_ribosomes/ribosomes/inputs")
        if v.endswith(".mrc")
    ]
    simulateExp(
        "data/scratch_data/cryodrgn_ribosomes/ribosomes",
        save_source=False,
        vols=ribo_vols,
        mask="data/scratch_data/cryodrgn_ribosomes/ribosomes/ribo_mask.mrc",
        L=128,
    )
    # [f'data/scratch_data/igg_1d/vols/128_org/{i:03}.mrc' for i in range(0,100,10)]
    # simulate_noisy_rots('data/pose_opt_exp_offsets_snr0.1',snr=0.1,rots_std = 0.1,offsets_std=0.008,r=5,
    #   vols = [f'data/scratch_data/igg_1d/vols/128_org/{int(i):03}.mrc' for i in np.linspace(0,100,6,endpoint=False)],
    #   mask='data/scratch_data/igg_1d/init_mask/mask.mrc')
