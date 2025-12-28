import copy
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from cov3d.covar import Covar, Mean
from cov3d.dataset import CovarDataset, GTData, create_dataloader, get_dataloader_batch_size
from cov3d.fsc_utils import (
    average_fourier_shell,
    covar_fsc,
    expand_fourier_shell,
    rpsd,
    upsample_and_expand_fourier_shell,
    vol_fsc,
)
from cov3d.mean import reconstruct_mean_from_halfsets, reconstruct_mean_from_halfsets_DDP
from cov3d.newton_opt import BlockwiseLBFGS
from cov3d.nufft_plan import NufftPlan, NufftPlanDiscretized
from cov3d.poses import (
    PoseModule,
    estimate_image_offsets_newton,
    in_plane_rot_error,
    offset_mean_error,
    out_of_plane_rot_error,
)
from cov3d.projection_funcs import (
    centered_fft2,
    centered_fft3,
    centered_ifft2,
    crop_image,
    get_mask_threshold,
    highpass_volume,
    lowpass_volume,
    preprocess_image_batch,
    vol_forward,
)
from cov3d.utils import cosineSimilarity, get_cpu_count, project_mean_out_from_eigenvecs, soft_edged_kernel
from cov3d.wiener_coords import compute_latentMAP_batch

logger = logging.getLogger(__name__)


class CovarTrainer:
    """Trainer class for covariance matrix estimation using SGD.

    Handles training of covariance models with various optimization strategies,
    logging, and evaluation metrics.

    Attributes:
        device: Device for computation (CPU or GPU)
        train_data: Training dataset or DataLoader
        _covar: Covariance model (may be wrapped in DDP)
        batch_size: Batch size for training
        isDDP: Whether using distributed data parallel
        save_path: Path to save training results
        logTraining: Whether to log training progress
        training_log_freq: Frequency of training logs
        gt_data: Ground truth data for evaluation
        training_log: Dictionary storing training metrics
        num_reduced_lr_before_stop: Number of LR reductions before stopping
        scheduler_patiece: Patience for learning rate scheduler
        apply_masking_on_epoch: Whether to apply masking during training
        fourier_reg: Fourier domain regularization term
    """

    def __init__(
        self,
        covar: Covar,
        train_data: Union[torch.utils.data.DataLoader, Any],
        device: torch.device,
        save_path: Optional[str] = None,
        gt_data: Optional[GTData] = None,
        training_log_freq: int = 50,
    ) -> None:
        self.device = device
        self.train_data = train_data
        self._covar = covar.to(device)

        self.batch_size = (
            get_dataloader_batch_size(train_data.data_iterable)
            if (not isinstance(train_data, torch.utils.data.DataLoader))
            else get_dataloader_batch_size(train_data)
        )
        self.isDDP = isinstance(self._covar, torch.nn.parallel.distributed.DistributedDataParallel)
        self.save_path = save_path
        self.logTraining = self.device.index == 0 or self.device == torch.device(
            "cpu"
        )  # Only log training on the first gpu
        self.training_log_freq = training_log_freq
        self.gt_data = gt_data
        if self.logTraining:
            self.training_log = {"epoch_ind": [], "lr_history": [], "cost_val": [], "epoch_run_time": []}
            if self.vectorsGT is not None:
                self.training_log.update(
                    {
                        "cosine_sim": [],
                        "fro_err": [],
                        "fro_err_half_res": [],
                        "fro_err_upper_half_res": [],
                        "fro_err_quarter_res": [],
                        "covar_fsc_mean": [],
                    }
                )

        self.num_reduced_lr_before_stop = 4
        self.scheduler_patiece = 0
        self.apply_masking_on_epoch = self.dataset.mask is not None
        self.fourier_reg = None

    @property
    def dataset(self) -> CovarDataset:
        """Get the underlying dataset from train_data.

        Returns:
            The dataset object
        """
        return (
            self.train_data.data_iterable.dataset
            if (not isinstance(self.train_data, torch.utils.data.DataLoader))
            else self.train_data.dataset
        )

    @property
    def dataloader_len(self) -> int:
        """Get the length of the dataloader.

        Returns:
            Number of batches in the dataloader
        """
        return (
            len(self.train_data.data_iterable)
            if (not isinstance(self.train_data, torch.utils.data.DataLoader))
            else len(self.train_data)
        )

    @property
    def covar(self) -> Covar:
        """Get the covariance model (unwrapped from DDP if needed).

        Returns:
            The covariance model
        """
        return self._covar.module if self.isDDP else self._covar

    @property
    def noise_var(self) -> float:
        """Get the noise variance from the dataset.

        Returns:
            Noise variance value
        """
        return self.dataset.noise_var

    @property
    def vectorsGT(self) -> Optional[torch.Tensor]:
        """Get ground truth vectors if available.

        Returns:
            Ground truth vectors or None
        """
        if self.gt_data is not None:
            if self.gt_data.eigenvecs is not None:
                return self.gt_data.eigenvecs
            else:
                return None
        else:
            return None

    @property
    def filter_gain(self) -> torch.Tensor:
        """Get the total covariance gain from the dataset.

        Returns:
            Filter gain tensor
        """
        return self.dataset.get_total_covar_gain(device=self.device)

    def to(self, *args: Any, **kwargs: Any) -> "CovarTrainer":
        """Move trainer to specified device.

        Args:
            *args: Positional arguments for torch.nn.Module.to
            **kwargs: Keyword arguments for torch.nn.Module.to

        Returns:
            Self for method chaining
        """
        self._covar = self._covar.to(*args, **kwargs)
        if self.fourier_reg is not None:
            self.fourier_reg = self.fourier_reg.to(*args, **kwargs)

        return self

    def run_batch(self, images: torch.Tensor, pts_rot: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """Run a single training batch.

        Args:
            images: Input images
            pts_rot: Rotated points for projection
            filters: CTF filters

        Returns:
            Cost value for the batch
        """
        self.nufft_plans.setpts(pts_rot)
        self.optimizer.zero_grad()
        cost_val = self.cost_func(
            self._covar(dummy_var=None),
            images,
            self.nufft_plans,
            filters,
            self.noise_var,
            self.reg_scale,
            self.fourier_reg,
        )  # Dummy_var is passed since for some reaosn DDP requires forward method to have an argument
        cost_val.backward()
        # torch.nn.utils.clip_grad_value_(self.covar.parameters(), 1e-3 * self.covar.grad_scale_factor)
        # #TODO : check for effect of gradient clipping
        self.optimizer.step()

        if self.use_orthogonal_projection:
            self.covar.orthogonal_projection()

        return cost_val

    def prepare_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for training.

        Args:
            batch: Raw batch data

        Returns:
            Tuple of (images, pts_rot, filters)
        """
        images, pts_rot, filters, _ = batch
        images = images.to(self.device)
        pts_rot = pts_rot.to(self.device)
        filters = filters.to(self.device)
        return images, pts_rot, filters

    def run_epoch(self, epoch: int) -> None:
        """Run a single training epoch.

        Args:
            epoch: Current epoch number
        """
        # if(self.isDDP):
        # self.train_data.sampler.set_epoch(epoch)
        if self.logTraining:
            pbar = tqdm(total=self.dataloader_len, desc=f"Epoch {epoch} , ", position=0, leave=True)

        mem_allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        mem_reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
        logger.debug(
            f"Device {self.device} GPU memory allocated: {mem_allocated:.2f} GB, reserved: {mem_reserved:.2f} GB"
        )

        self.cost_in_epoch = torch.tensor(0, device=self.device, dtype=torch.float32)
        for batch_ind, data in enumerate(self.train_data):
            images, pts_rot, filters = self.prepare_batch(data)
            cost_val = self.run_batch(images, pts_rot, filters)
            with torch.no_grad():
                self.cost_in_epoch += cost_val * self.batch_size

            if self.logTraining:
                if batch_ind % self.training_log_freq == 0:
                    self.log_training(epoch, batch_ind, cost_val)
                    pbar.set_description(self._get_pbar_desc(epoch))

                pbar.update(1)

        if self.isDDP:
            torch.distributed.all_reduce(self.cost_in_epoch, op=torch.distributed.ReduceOp.SUM)

        if self.logTraining:
            logger.debug("Total cost value in epoch : {:.2e}".format(self.cost_in_epoch.item()))

    def get_trainable_parameters(self) -> List[Dict[str, Union[torch.nn.Parameter, float]]]:
        """Get trainable parameters with learning rate factors.

        Returns:
            List of parameter groups with learning rates
        """
        return self.covar.grad_lr_factor()

    def train(self, max_epochs: int, **training_kwargs: Any) -> None:
        """Train the model for specified number of epochs.

        Args:
            max_epochs: Maximum number of epochs to train
            **training_kwargs: Additional training parameters
        """
        self.setup_training(**training_kwargs)
        self.train_epochs(max_epochs)
        self.complete_training()

    def setup_training(
        self,
        lr: Optional[float] = None,
        momentum: float = 0.9,
        optim_type: str = "Adam",
        reg: float = 1,
        nufft_disc: str = "bilinear",
        orthogonal_projection: bool = False,
        objective_func: str = "ml",
    ) -> None:
        """Setup training configuration.

        Args:
            lr: Learning rate (if None, uses default based on optimizer)
            momentum: Momentum for SGD optimizer
            optim_type: Type of optimizer ("Adam" or "SGD")
            reg: Regularization scaling factor
            nufft_disc: NUFFT discretization method
            orthogonal_projection: Whether to use orthogonal projection
            objective_func: Objective function type ("ml" or "ls")
        """
        self.use_orthogonal_projection = orthogonal_projection

        if lr is None:
            lr = 1e-1 if optim_type == "Adam" else 1e-2  # Default learning rate for Adam/SGD optimizer

        lr *= self.batch_size if not self.isDDP else self.batch_size * torch.distributed.get_world_size()
        self.lr = lr
        self.optim_type = optim_type
        self.momentum = momentum
        self.restart_optimizer()

        rank = self.covar.rank
        dtype = self.covar.dtype
        vol_shape = (self.covar.resolution,) * 3

        # When NUFFT discretization is used - we optimize in Fourier domain
        self.optimize_in_fourier_domain = nufft_disc != "nufft"

        self.objective_func = objective_func
        if self.optimize_in_fourier_domain:
            self.nufft_plans = NufftPlanDiscretized(
                vol_shape, upsample_factor=self.covar.upsampling_factor, mode=nufft_disc
            )
            self.dataset.to_fourier_domain()
            self.cost_func = cost_fourier_domain if objective_func == "ls" else cost_maximum_liklihood_fourier_domain
        else:
            self.nufft_plans = NufftPlan(vol_shape, batch_size=rank, dtype=dtype, device=self.device)
            self.dataset.to_spatial_domain()
            self.cost_func = cost if objective_func == "ls" else cost_maximum_liklihood
        self.covar.init_grid_correction(nufft_disc)
        logger.debug(f"Actual learning rate {lr}")

        # The sgd is performed on cost/batch_size + reg_term while its supposed to be sum(cost) + reg_term.
        # This ensures the regularization term scales in the appropriate manner
        self.reg_scale = reg / (len(self.dataset))
        self.epoch_index = 0

    def restart_optimizer(self) -> None:
        """Restart the optimizer with current parameters.

        Creates new optimizer and scheduler instances.
        """
        params_lr = self.get_trainable_parameters()
        for i in range(len(params_lr)):
            params_lr[i]["lr"] *= self.lr

        if self.optim_type == "SGD":
            self.optimizer = torch.optim.SGD(params_lr, momentum=self.momentum)
        elif self.optim_type == "Adam":
            self.optimizer = torch.optim.Adam(params_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.scheduler_patiece)

    def train_epochs(self, max_epochs: int, restart_optimizer: bool = False) -> None:
        """Train for multiple epochs.

        Args:
            max_epochs: Maximum number of epochs to train
            restart_optimizer: Whether to restart optimizer before training
        """
        if restart_optimizer:
            self.restart_optimizer()

        for epoch in range(max_epochs):
            epoch_start_time = time.time()
            self.run_epoch(self.epoch_index)
            epoch_end_time = time.time()
            logger.info(f"Epoch runtime: {epoch_end_time - epoch_start_time:.2f} seconds")

            self.scheduler.step(self.cost_in_epoch)
            logger.debug(f"New learning rate set to {self.scheduler.get_last_lr()}")

            # Apply masking on covar vectors
            if self.apply_masking_on_epoch:
                with torch.no_grad():
                    mask = self.dataset.mask.to(self.device) > 0.3
                    self.covar.vectors.data.copy_(self.covar.vectors.data * mask)

            if self.logTraining and self.save_path is not None:
                self.training_log["epoch_run_time"].append(epoch_end_time - epoch_start_time)
                self.save_result()

            self.epoch_index += 1
            if self.scheduler.get_last_lr()[0] <= self.lr * (self.scheduler.factor**self.num_reduced_lr_before_stop):
                logger.debug(
                    f"Learning rate has been reduced {self.num_reduced_lr_before_stop} times. Stopping training."
                )
                break

    def complete_training(self) -> None:
        """Complete training and cleanup.

        Performs final cleanup operations after training.
        """
        torch.cuda.empty_cache()
        if self.optimize_in_fourier_domain:  # Transform back to spatial domain
            self.dataset.to_spatial_domain()

    def compute_fourier_reg_term(self, eigenvecs: torch.Tensor) -> None:
        """Compute Fourier domain regularization term from an estimate of the covariance eigen
        vectors.

        Args:
            eigenvecs: Covariance eigen vectors estimate
        """
        eigen_rpsd = rpsd(*eigenvecs) * (self.covar.upsampling_factor**3)
        self.fourier_reg = (self.noise_var) / upsample_and_expand_fourier_shell(
            eigen_rpsd, self.covar.resolution * self.covar.upsampling_factor, 3
        )

    def update_fourier_reg_halfsets(self, fourier_reg: torch.Tensor) -> None:
        """Update Fourier regularization from half-sets.

        Args:
            fourier_reg: Fourier regularization tensor
        """
        fourier_reg = fourier_reg.to(self.device)

        if self.optimize_in_fourier_domain:
            # This ensures that the fourier_reg term is in the same as the upsampled size of covar
            fourier_reg_radial = average_fourier_shell(fourier_reg) / (self.covar.upsampling_factor**3)
            fourier_reg = upsample_and_expand_fourier_shell(
                fourier_reg_radial, self.covar.resolution * self.covar.upsampling_factor, 3
            )

        self.fourier_reg = fourier_reg

    def log_training(self, num_epoch: int, batch_ind: int, cost_val: torch.Tensor) -> None:
        """Log training metrics.

        Args:
            num_epoch: Current epoch number
            batch_ind: Current batch index
            cost_val: Current cost value
        """
        self.training_log["epoch_ind"].append(num_epoch + batch_ind / self.dataloader_len)
        self.training_log["lr_history"].append(self.scheduler.get_last_lr()[0])
        self.training_log["cost_val"].append(cost_val.detach().cpu().numpy())

        if self.vectorsGT is not None:
            with torch.no_grad():
                L = self.covar.resolution
                vectors = self.covar.get_vectors_spatial_domain()
                vectorsGT = self.vectorsGT.to(self.device).reshape(self.vectorsGT.shape[0], L, L, L)
                self.training_log["covar_fsc_mean"].append(
                    (covar_fsc(vectorsGT, vectors)[: L // 2, : L // 2].mean().cpu().numpy())
                )
                self.training_log["fro_err"].append(
                    (frobeniusNormDiff(vectorsGT, vectors) / frobeniusNorm(vectorsGT)).cpu().numpy()
                )

                vectors_lowpass = lowpass_volume(vectors, L // 4)
                vectors_GT_lowpass = lowpass_volume(vectorsGT, L // 4)
                self.training_log["fro_err_half_res"].append(
                    (frobeniusNormDiff(vectors_GT_lowpass, vectors_lowpass) / frobeniusNorm(vectors_GT_lowpass))
                    .cpu()
                    .numpy()
                )

                vectors_highpass = highpass_volume(vectors, L // 4)
                vectors_GT_highpass = highpass_volume(vectorsGT, L // 4)
                self.training_log["fro_err_upper_half_res"].append(
                    (frobeniusNormDiff(vectors_GT_highpass, vectors_highpass) / frobeniusNorm(vectors_GT_highpass))
                    .cpu()
                    .numpy()
                )

                vectors_lowpass = lowpass_volume(vectors, L // 8)
                vectors_GT_lowpass = lowpass_volume(vectorsGT, L // 8)
                self.training_log["fro_err_quarter_res"].append(
                    (frobeniusNormDiff(vectors_GT_lowpass, vectors_lowpass) / frobeniusNorm(vectors_GT_lowpass))
                    .cpu()
                    .numpy()
                )

                vectors = vectors.reshape((vectors.shape[0], -1))
                vectorsGT = vectorsGT.reshape((vectorsGT.shape[0], -1))
                self.training_log["cosine_sim"].append(cosineSimilarity(vectors.detach(), vectorsGT))

    def _get_pbar_desc(self, epoch: int) -> str:
        """Get progress bar description.

        Args:
            epoch: Current epoch number

        Returns:
            Progress bar description string
        """
        pbar_description = f"Epoch {epoch} , " + "cost value : {:.2e}".format(self.cost_in_epoch)
        pbar_description += f" , vecs norm : {torch.norm(self.covar.get_vectors())}"
        if self.vectorsGT is not None:
            # TODO : update log metrics, use principal angles
            cosine_sim_val = np.mean(np.sqrt(np.sum(self.training_log["cosine_sim"][-1] ** 2, axis=0)))
            fro_err_val = self.training_log["fro_err"][-1]
            pbar_description = (
                pbar_description
                + ",  cosine sim : {:.2f}".format(cosine_sim_val)
                + ", frobenium norm error : {:.2e}".format(fro_err_val)
                + ", covar fsc mean : {:.2e}".format(self.training_log["covar_fsc_mean"][-1])
            )
        return pbar_description

    def results_dict(self) -> Dict[str, Any]:
        """Get results dictionary for saving.

        Returns:
            Dictionary containing training results
        """
        ckp = self.covar.state_dict()
        ckp["vectorsGT"] = self.vectorsGT
        ckp["fourier_reg"] = self.fourier_reg
        ckp.update(self.training_log)

        return ckp

    def save_result(self) -> None:
        """Save training results to file.

        Saves the current state and training log to the specified path.
        """
        savedir = os.path.split(self.save_path)[0]
        os.makedirs(savedir, exist_ok=True)
        ckp = self.results_dict()
        torch.save(ckp, self.save_path)


class CovarPoseTrainer(CovarTrainer):
    """Trainer class for covariance matrix estimation with pose optimization.

    Extends CovarTrainer to include pose estimation and mean volume optimization.

    Attributes:
        mean: Mean volume model
        pose: Pose module for rotation and translation estimation
        pose_lr_ratio: Learning rate ratio for pose optimization
        num_rep: Number of repetitions for pose optimization
        _updated_idx: Track which poses have been updated
        mean_fourier_reg: Fourier regularization for mean volume
        scheduler_patiece: Patience for learning rate scheduler
        downsample_factor: Factor for downsampling during training
        mean_est_method: Method for mean estimation
        offset_est_method: Method for offset estimation
        rotation_est_method: Method for rotation estimation
        mean_update_frequency: Frequency of mean volume updates
    """

    def __init__(
        self,
        covar: Covar,
        train_data: Union[torch.utils.data.DataLoader, Any],
        device: torch.device,
        mean: Mean,
        pose: PoseModule,
        save_path: Optional[str] = None,
        gt_data: Optional[GTData] = None,
        training_log_freq: int = 50,
    ) -> None:
        super().__init__(covar, train_data, device, save_path, gt_data, training_log_freq)
        self.mean = mean.to(self.device)
        self.pose = pose.to(self.device)
        self.pose_lr_ratio = 3
        self.num_rep = 1
        self.set_pose_grad_req(True)
        self._updated_idx = torch.zeros(len(self.dataset), device=self.device)

        if self.logTraining and self.gt_data is not None:
            if self.gt_data.rotations is not None:
                self.training_log.update({"rot_angle_dist": [], "in_plane_rot_angle_dist": []})
            if self.gt_data.offsets is not None:
                self.training_log.update({"offsets_mean_dist": []})
            if self.gt_data.contrasts is not None and self.get_pose_module().use_contrast:
                self.training_log.update({"contrast_mean_dist": [], "contrast_corr": []})
            if self.gt_data.mean is not None:
                self.training_log.update({"mean_vol_norm_err": [], "mean_vol_fsc": []})

        self.mean_fourier_reg = None
        self.scheduler_patiece = 3
        self.downsample_factor = min(3, int(np.log2(self.covar.resolution / 32)))
        self.mean_est_method = "Reconstruction"
        self.offset_est_method = "Newton"
        self.rotation_est_method = "SGD"
        self.mean_update_frequency = 5
        self.use_eigen_in_contrast_from_epoch = 0

        if self.mean_est_method != "SGD":
            for param in self.get_mean_module().parameters():
                param.requires_grad = False
        # for param in self.covar.parameters():
        #    param.requires_grad = False
        # for param in self.get_pose_module().parameters():
        #    param.requires_grad = False
        if self.offset_est_method != "SGD":
            for param in self.get_pose_module().offsets.parameters():
                param.requires_grad = False
            if self.get_pose_module().use_contrast:
                for param in self.get_pose_module().contrasts.parameters():
                    param.requires_grad = False

    @property
    def downsample_size(self) -> int:
        """Get the downsampled size for pose optimization.

        Returns:
            Downsampled resolution
        """
        return int(2 ** (-self.downsample_factor) * self.pose.resolution)

    def get_mean_module(self) -> Mean:
        """Get the mean module (unwrapped from DDP if needed).

        Returns:
            The mean module
        """
        return self.mean if not self.isDDP else self.mean.module

    def get_pose_module(self) -> PoseModule:
        """Get the pose module.

        Returns:
            The pose module
        """
        return self.pose

    def set_pose_module(self, pose: PoseModule) -> None:
        """Set the pose module.

        Args:
            pose: New pose module
        """
        self.pose = pose.to(self.device)
        if self.offset_est_method != "SGD":
            for param in self.get_pose_module().offsets.parameters():
                param.requires_grad = False
            if self.get_pose_module().use_contrast:
                for param in self.get_pose_module().contrasts.parameters():
                    param.requires_grad = False
        self._updated_idx = torch.zeros(len(self.dataset), device=self.device)

    def _ddp_sync_pose_module(self) -> None:
        """Synchronize pose module across distributed processes.

        Manually synchronizes pose parameters since DDP doesn't support sparse gradients.
        """
        if not self.isDDP:
            return
        # For some reason DDP does support sparse gradients.
        # Hence we do not wrap the pose module in DDP and have to sync it manually

        updated_idx_list = [torch.zeros_like(self._updated_idx) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(updated_idx_list, self._updated_idx)

        rotvecs = self.get_pose_module().get_rotvecs()
        rotvec_list = [torch.zeros_like(rotvecs) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(rotvec_list, rotvecs)

        offsets = self.get_pose_module().get_offsets()
        offsets_list = [torch.zeros_like(offsets) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(offsets_list, offsets)

        if self.get_pose_module().use_contrast:
            contrasts = self.get_pose_module().get_contrasts()
            contrasts_list = [torch.zeros_like(contrasts) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(contrasts_list, contrasts)

        idx_to_update = torch.zeros_like(self._updated_idx)
        for i in range(len(updated_idx_list)):
            # If idx_to_update is already set to 1, we don't need to update it again
            # this means that a previous node has already updated this index
            # (can happen when len(dataset) % world_size != 0)
            updated_idx_list[i][idx_to_update > 0] = 0

            idx_to_update[updated_idx_list[i] > 0] = 1
            # rotvecs and offsets point to the same memory location in the pose module - so we can just update them here
            rotvecs[updated_idx_list[i] > 0] = rotvec_list[i][updated_idx_list[i] > 0]
            offsets[updated_idx_list[i] > 0] = offsets_list[i][updated_idx_list[i] > 0]
            if self.get_pose_module().use_contrast:
                contrasts[updated_idx_list[i] > 0] = contrasts_list[i][updated_idx_list[i] > 0]

        # Reset updated_idx
        self._updated_idx = torch.zeros_like(self._updated_idx)

    def correct_offsets(self) -> None:
        """Correct image offsets using Newton optimization.

        Iterates through batches and estimates optimal offsets for each image.
        """
        for batch in tqdm(self.train_data, desc="Correcting offsets"):
            images, idx, filters = self.prepare_batch(batch)

            softening_kernel_fourier = self.softening_kernel_fourier
            if self.downsample_size != self.pose.resolution:
                images = crop_image(images, self.downsample_size)
                filters = crop_image(filters, self.downsample_size)
                softening_kernel_fourier = crop_image(softening_kernel_fourier, self.downsample_size)

            self.nufft_plans.setpts(self.pose(idx, ds_resolution=self.downsample_size)[0].detach())
            mean_forward = (
                vol_forward(
                    self.mean(dummy_var=None),
                    self.nufft_plans,
                    filters=filters,
                    fourier_domain=self.optimize_in_fourier_domain,
                )
                .squeeze(1)
                .detach()
            )

            mask_forward = vol_forward(self.mask, self.nufft_plans, filters=None, fourier_domain=False).squeeze(1)
            mask_forward = mask_forward > self.mask_threshold
            soft_mask = centered_ifft2(centered_fft2(mask_forward) * softening_kernel_fourier).real

            projected_eigenvecs = vol_forward(
                self._covar(dummy_var=None).detach(),
                self.nufft_plans,
                filters,
                fourier_domain=self.optimize_in_fourier_domain,
            )

            if self.get_pose_module().use_contrast:
                shifted_images = images * self.pose(idx, ds_resolution=self.downsample_size)[1]

                predicted_images = mean_forward
                if self.epoch_index >= self.use_eigen_in_contrast_from_epoch:
                    latent_coords, _, _ = compute_latentMAP_batch(
                        shifted_images - mean_forward, projected_eigenvecs, self.dataset.noise_var
                    )
                    predicted_images += torch.sum(projected_eigenvecs * latent_coords.unsqueeze(-1), dim=1)

                contrast_est = (
                    torch.sum(predicted_images.conj() * shifted_images, dim=(-1, -2))
                    / torch.norm(predicted_images, dim=(-1, -2)) ** 2
                ).real
                # contrast_est = contrast_est.clamp(0.5,1.5)
                self.get_pose_module().set_contrasts(contrast_est.unsqueeze(-1), idx=idx)

            def obj_func(phase_shifted_image):
                if not self.optimize_in_fourier_domain:
                    phase_shifted_image = centered_ifft2(phase_shifted_image).real
                preprocessed_images = phase_shifted_image - mean_forward

                return self.raw_cost_func(
                    projected_eigenvecs,
                    preprocessed_images,
                    self.noise_var,
                    apply_mean_const_term=True,
                    mean_aggregate=False,
                )

            offsets = estimate_image_offsets_newton(
                images,
                mean_forward,
                mask=soft_mask,
                init_offsets=self.get_pose_module().get_offsets()[idx].detach(),
                in_fourier_domain=self.optimize_in_fourier_domain,
                obj_func=obj_func,
            )

            self.get_pose_module().set_offsets(offsets, idx=idx)

        if self.get_pose_module().use_contrast:
            normalized_contrast = self.get_pose_module().get_contrasts()
            normalized_contrast /= torch.mean(normalized_contrast)
            self.get_pose_module().set_contrasts(normalized_contrast)

    def set_pose_grad_req(self, grad: bool) -> None:
        """Set gradient requirements for pose and mean modules.

        Args:
            grad: Whether gradients are required
        """
        for param in self.get_pose_module().parameters():
            param.requires_grad = grad
        for param in self.get_mean_module().parameters():
            param.requires_grad = grad

    def get_trainable_parameters(self) -> List[Dict[str, Union[torch.nn.Parameter, float]]]:
        """Get trainable parameters including mean module.

        Returns:
            List of parameter groups with learning rates
        """
        trainable_params = super().get_trainable_parameters() + [
            {"params": self.mean.parameters(), "lr": 1},
        ]
        return trainable_params

    def prepare_batch(self, batch: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare batch data for pose training.

        Args:
            batch: Raw batch data

        Returns:
            Tuple of (images, idx, filters)
        """
        images, _, filters, idx = batch
        images = images.to(self.device)
        idx = idx.to(self.device)
        filters = filters.to(self.device)
        return images, idx, filters

    def run_batch(self, images: torch.Tensor, idx: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
        """Run a single training batch with pose optimization.

        Args:
            images: Input images
            idx: Image indices
            filters: CTF filters

        Returns:
            Cost value for the batch
        """
        for _ in range(self.num_rep):
            self.optimizer.zero_grad()
            self.pose_optimizer.zero_grad()

            # Currenly no actual downsampling is taking place
            downsample_size = self.covar.resolution

            if self.get_pose_module().use_contrast:
                # If we optimize over contrasts scale filters
                pts_rot, phase_shift, contrasts = self.pose(idx, ds_resolution=downsample_size)
                filters = filters * contrasts.reshape(-1, 1, 1)
            else:
                pts_rot, phase_shift = self.pose(idx, ds_resolution=downsample_size)

            softening_kernel_fourier = self.softening_kernel_fourier
            if downsample_size != self.pose.resolution:
                images = crop_image(images, downsample_size)
                filters = crop_image(filters, downsample_size)
                softening_kernel_fourier = crop_image(softening_kernel_fourier, downsample_size)

            preprocessed_images = preprocess_image_batch(
                images,
                self.nufft_plans,
                filters,
                (pts_rot, phase_shift),
                self.mean(dummy_var=None),
                self.mask,
                self.mask_threshold,
                softening_kernel_fourier,
                fourier_domain=self.optimize_in_fourier_domain,
            )

            cost_val = self.cost_func(
                self._covar(dummy_var=None),
                preprocessed_images,
                self.nufft_plans,
                filters,
                self.noise_var,
                self.reg_scale,
                self.fourier_reg,
                apply_mean_const_term=True,
            )  # Dummy_var is passed since for some reaosn DDP requires forward method to have an argument
            if self.mean_est_method == "SGD":
                cost_val = cost_val + self.regularize_mean()
            cost_val.backward()

            self.optimizer.step()
            self.pose_optimizer.step()

        if self.use_orthogonal_projection:
            self.covar.orthogonal_projection()

        with torch.no_grad():
            lowpassed_vecs = lowpass_volume(self.covar.vectors.data, self.downsample_size // 2)
            self.covar.vectors.data.copy_(lowpassed_vecs)

        self._updated_idx[idx] = 1

        return cost_val

    def run_epoch(self, epoch: int) -> None:
        """Run a single training epoch with pose optimization.

        Args:
            epoch: Current epoch number
        """
        super().run_epoch(epoch)

        if self.get_pose_module().use_contrast:
            with torch.no_grad():
                projected_vecs = project_mean_out_from_eigenvecs(
                    self.covar.get_vectors_spatial_domain(), self.mean.get_volume_spatial_domain()
                )
                self.covar.vectors.data.copy_(
                    projected_vecs / torch.exp(self.covar.log_sqrt_eigenvals).reshape(-1, 1, 1, 1)
                )

        if epoch % self.mean_update_frequency == self.mean_update_frequency - 1:
            if self.offset_est_method == "Newton":
                self.correct_offsets()
            self._ddp_sync_pose_module()

            if self.mean_est_method == "Reconstruction":
                # update datatset pts_rot and update the mean volume estimate

                # We can use larger batch size since this computation is very light weight
                batch_size = self.batch_size * 16

                # TODO: this is in efficient when DDP is used.
                # This is because in that case each node only uses a fraction of the dataset
                # and we don't have to update all indeces of pts_rot.
                self.dataset.update_pose(self.get_pose_module(), batch_size=batch_size)

                if not self.isDDP:
                    reconstructed_mean = reconstruct_mean_from_halfsets(
                        self.dataset, mask=self.get_mean_module().volume_mask
                    )
                else:
                    reconstructed_mean = reconstruct_mean_from_halfsets_DDP(
                        self.train_data, ranks=self._covar.ranks_in_group, mask=self.get_mean_module().volume_mask
                    )
                self.get_mean_module().set_mean(reconstructed_mean)

    def setup_training(self, **kwargs: Any) -> None:
        """Setup training configuration for pose optimization.

        Args:
            **kwargs: Additional training parameters
        """
        super().setup_training(**kwargs)
        self.get_mean_module().init_grid_correction(kwargs.get("nufft_disc"))
        self.mask = self.get_mean_module().get_volume_mask()
        self.raw_cost_func = (
            raw_cost_fourier_domain
            if kwargs.get("objective_func") == "ls"
            else raw_cost_maximum_liklihood_fourier_domain
        )

        self.softening_kernel_fourier = (
            soft_edged_kernel(radius=5, L=self.get_mean_module().resolution, dim=2, in_fourier=True)
            .to(self.device)
            .to(self.mask.dtype)
        )

        with torch.no_grad():
            idx = torch.arange(self.batch_size, device=self.device)
            self.nufft_plans.setpts(self.pose(idx)[0])
            self.mask_threshold = get_mask_threshold(self.mask, self.nufft_plans)

    def complete_training(self) -> None:
        """Complete training and cleanup for pose optimization.

        Performs final cleanup operations after training including pose synchronization.
        """
        super().complete_training()
        self._ddp_sync_pose_module()

    def restart_optimizer(self) -> None:
        """Restart the optimizer with current parameters including pose optimizer.

        Creates new optimizer and scheduler instances for both covariance and pose optimization.
        """
        super().restart_optimizer()
        if self.rotation_est_method == "SGD":
            self.pose_optimizer = torch.optim.SparseAdam(
                [{"params": self.pose.parameters(), "lr": self.pose_lr_ratio * self.lr}]
            )
        elif self.rotation_est_method == "LBFGS":
            self.pose_optimizer = BlockwiseLBFGS([{"params": self.pose.parameters(), "lr": 1, "step_size_limit": 1e-1}])

    def train_epochs(self, max_epochs: int, **training_kwargs: Any) -> None:
        """Train for multiple epochs with progressive downsampling.

        Args:
            max_epochs: Maximum number of epochs to train
            **training_kwargs: Additional training parameters
        """
        while self.downsample_factor >= 0:
            logger.debug(f"Using downsample factor of {self.downsample_factor}")
            self.apply_masking_on_epoch = self.downsample_factor == 0
            super().train_epochs(max_epochs, **training_kwargs)
            self.downsample_factor -= 1
        self.downsample_factor = 0

    def compute_fourier_reg_term(self, eigenvecs: Tuple[torch.Tensor, torch.Tensor]) -> None:
        """Compute Fourier domain regularization term for pose optimization.

        Args:
            eigenvecs: Tuple of (eigenvectors, eigenvalues)
        """
        super().compute_fourier_reg_term(eigenvecs)
        if self.mean_est_method == "SGD":
            self.compute_fourier_mean_reg_term()

    def compute_fourier_mean_reg_term(self) -> None:
        """Compute Fourier regularization term for mean volume.

        Computes regularization based on the power spectral density of the mean volume.
        """
        mean = self.get_mean_module()
        mean_rpsd = rpsd(*mean.get_volume_spatial_domain().detach())
        mean_rpsd[-1] = mean_rpsd[-2]  # TODO: fix tail of rpsd
        self.mean_fourier_reg = 1 / upsample_and_expand_fourier_shell(
            mean_rpsd, mean.resolution * mean.upsampling_factor, 3
        )

    def regularize_mean(self) -> torch.Tensor:
        """Compute regularization cost for mean volume.

        Returns:
            Regularization cost tensor
        """
        if self.mean_fourier_reg is None:
            return 0

        if self.objective_func == "ml":
            vol_fourier = self.get_mean_module().get_volume_fourier_domain() * torch.sqrt(self.mean_fourier_reg)
            reg_cost = torch.norm(vol_fourier) ** 2 * self.reg_scale
            return reg_cost

        else:
            raise Exception("Mean regularization not implemented for this objective function")

    def log_training(self, num_epoch: int, batch_ind: int, cost_val: torch.Tensor) -> None:
        """Log training metrics including pose errors.

        Args:
            num_epoch: Current epoch number
            batch_ind: Current batch index
            cost_val: Current cost value
        """
        from aspire.utils import Rotation

        super().log_training(num_epoch, batch_ind, cost_val)
        if self.gt_data is not None:
            with torch.no_grad():
                if self.gt_data.rotations is not None:
                    rotvecs = self.get_pose_module().get_rotvecs()
                    rots_est = Rotation.from_rotvec(rotvecs.cpu().numpy())
                    rots_gt = Rotation(self.gt_data.rotations.numpy())
                    angle_diff = out_of_plane_rot_error(torch.tensor(rots_est.matrices), torch.tensor(rots_gt.matrices))
                    in_plane_angle_diff = in_plane_rot_error(
                        torch.tensor(rots_est.matrices), torch.tensor(rots_gt.matrices)
                    )
                    self.training_log["rot_angle_dist"].append(angle_diff[1])
                    self.training_log["in_plane_rot_angle_dist"].append(in_plane_angle_diff[1])

                if self.gt_data.offsets is not None:
                    offsets_est = self.get_pose_module().get_offsets()
                    offsets_gt = self.gt_data.offsets
                    self.training_log["offsets_mean_dist"].append(
                        offset_mean_error(offsets_est.cpu(), offsets_gt, L=self.covar.resolution)
                    )
                if self.gt_data.mean is not None:
                    mean_gt = self.gt_data.mean.to(self.device)
                    mean_est = self.get_mean_module().get_volume_spatial_domain()
                    mean_fsc = vol_fsc(mean_gt, mean_est.squeeze(0))
                    mean_fsc = mean_fsc[: mean_gt.shape[-1] // 2].mean()
                    self.training_log["mean_vol_norm_err"].append(
                        torch.norm(mean_gt - mean_est).cpu().numpy() / torch.norm(mean_gt).cpu().numpy()
                    )
                    self.training_log["mean_vol_fsc"].append(mean_fsc.cpu().numpy())
                if self.gt_data.contrasts is not None and self.get_pose_module().use_contrast:
                    cont = self.get_pose_module().get_contrasts()
                    cont_gt = self.gt_data.contrasts
                    cont_mean_err = torch.norm(cont.cpu() - cont_gt, dim=1).mean().numpy()
                    cont_corr = torch.corrcoef(torch.concat([cont.cpu(), cont_gt.cpu()], dim=1).T)[0, 1].numpy()
                    self.training_log["contrast_mean_dist"].append(cont_mean_err)
                    self.training_log["contrast_corr"].append(cont_corr)

    def _get_pbar_desc(self, epoch: int) -> str:
        """Get progress bar description with pose information.

        Args:
            epoch: Current epoch number

        Returns:
            Progress bar description string
        """
        pbar_description = super()._get_pbar_desc(epoch)
        if self.gt_data is not None:
            if self.gt_data.rotations is not None:
                pbar_description += (
                    f" , mean angle dist out-of-plane: "
                    f"{self.training_log['rot_angle_dist'][-1]:.2e} , "
                    f"in-plane: {self.training_log['in_plane_rot_angle_dist'][-1]:.2e}"
                )
            if self.gt_data.offsets is not None:
                pbar_description += f" , offset mean : {self.training_log['offsets_mean_dist'][-1]:.2e}"
            if self.gt_data.contrasts is not None and self.get_pose_module().use_contrast:
                pbar_description += f" , contrast corr: {self.training_log['contrast_corr'][-1]:.2e}"
            if self.gt_data.mean is not None:
                pbar_description += f" , mean vol norm err : {self.training_log['mean_vol_norm_err'][-1]:.2e}"
                pbar_description += f" , mean vol fsc : {self.training_log['mean_vol_fsc'][-1]:.2e}"
        return pbar_description

    def results_dict(self) -> Dict[str, Any]:
        """Get results dictionary for saving with pose information.

        Returns:
            Dictionary containing training results including pose and mean states
        """
        ckp = super().results_dict()
        ckp["mean"] = self.get_mean_module().state_dict()
        ckp["pose"] = self.get_pose_module().state_dict()
        return ckp


def update_fourier_reg(trainer1: CovarTrainer, trainer2: CovarTrainer) -> None:
    """Update Fourier regularization using half-sets.

    Args:
        trainer1: First trainer (half-set 1)
        trainer2: Second trainer (half-set 2)
    """
    L = trainer1.covar.resolution
    filter_gain = (trainer1.filter_gain + trainer2.filter_gain) / 2
    current_fourier_reg = trainer1.fourier_reg
    # Get the covariance eigenvectors from each trainer
    eigenvecs1 = trainer1.covar.eigenvecs
    eigenvecs1 = eigenvecs1[0] * (eigenvecs1[1] ** 0.5).reshape(-1, 1, 1, 1)

    eigenvecs2 = trainer2.covar.eigenvecs
    eigenvecs2 = eigenvecs2[0] * (eigenvecs2[1] ** 0.5).reshape(-1, 1, 1, 1)

    new_fourier_reg_tensor, covariance_fsc = compute_updated_fourier_reg(
        eigenvecs1,
        eigenvecs2,
        filter_gain,
        current_fourier_reg,
        L,
        trainer1.optimize_in_fourier_domain,
        mask=trainer1.dataset.mask,
    )

    trainer1.update_fourier_reg_halfsets(new_fourier_reg_tensor)
    trainer2.update_fourier_reg_halfsets(new_fourier_reg_tensor)

    if trainer1.logTraining:
        trainer1.training_log["covariance_fsc_halfset"] = covariance_fsc


def compute_updated_fourier_reg(
    eigenvecs1, eigenvecs2, filter_gain, current_fourier_reg, L, optimize_in_fourier_domain, mask=None
):

    if current_fourier_reg is None:
        current_fourier_reg = torch.zeros((L,) * 3, dtype=filter_gain.dtype, device=filter_gain.device)

    filter_gain_shell_correction = 1 / filter_gain
    filter_gain_shell_correction[:, L // 2 :] = filter_gain_shell_correction[L // 2, L // 2]
    filter_gain_shell_correction[L // 2 :, :] = filter_gain_shell_correction[L // 2, L // 2]

    if mask is not None:
        mask = mask.clone().to(eigenvecs1.device)
        eigenvecs1 = eigenvecs1 * mask
        eigenvecs2 = eigenvecs2 * mask

    covariance_fsc = covar_fsc(eigenvecs1, eigenvecs2)
    fsc_epsilon = 1e-6
    covariance_fsc[covariance_fsc < fsc_epsilon] = fsc_epsilon
    covariance_fsc[covariance_fsc > 1 - fsc_epsilon] = 1 - fsc_epsilon

    new_fourier_reg = 1 / ((covariance_fsc / (1 - covariance_fsc)) * filter_gain_shell_correction)
    new_fourier_reg[new_fourier_reg < 0] = 0

    # This is a heuristic approach to get a rank 1 approx of the 'regulariztaion matrix'
    # which allows much faster computation of the regularizaiton term
    new_fourier_reg = expand_fourier_shell(new_fourier_reg.diag().sqrt().unsqueeze(0), L, 3)

    if not optimize_in_fourier_domain:
        # When optimizing in spatial domain regularization needs to be scaled by L^2
        new_fourier_reg /= L**2

    return new_fourier_reg, covariance_fsc


def cost(
    vols,
    images,
    nufft_plans,
    filters,
    noise_var,
    reg_scale=0,
    fourier_reg=None,
    apply_mean_const_term=False,
    mean_aggregate=True,
):
    batch_size = images.shape[0]
    rank = vols.shape[0]
    L = vols.shape[-1]
    projected_vols = vol_forward(vols, nufft_plans, filters)

    images = images.reshape((batch_size, 1, -1))
    projected_vols = projected_vols.reshape((batch_size, rank, -1))

    images_projvols_term = torch.matmul(projected_vols, images.transpose(1, 2))
    projvols_prod_term = torch.matmul(projected_vols, projected_vols.transpose(1, 2))

    cost_val = -2 * torch.sum(torch.pow(images_projvols_term, 2), dim=(1, 2)) + torch.sum(
        torch.pow(projvols_prod_term, 2), dim=(1, 2)
    )

    # Add noise cost terms
    norm_squared_projvols = torch.diagonal(projvols_prod_term, dim1=1, dim2=2)
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols, dim=1))

    if apply_mean_const_term:
        norm_squared_images = torch.pow(torch.norm(images, dim=(1, 2)), 2)
        cost_val += torch.pow(norm_squared_images, 2)
        cost_val -= 2 * noise_var * norm_squared_images + (noise_var * L) ** 2

    cost_val = torch.mean(cost_val, dim=0) if mean_aggregate else cost_val

    if fourier_reg is not None and reg_scale != 0:
        vols_fourier = centered_fft3(vols)
        vols_fourier *= torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank, -1))
        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(), 2))
        cost_val += reg_scale * reg_cost

    return cost_val


def cost_maximum_liklihood(
    vols,
    images,
    nufft_plans,
    filters,
    noise_var,
    reg_scale=0,
    fourier_reg=None,
    apply_mean_const_term=False,
    mean_aggregate=True,
):
    batch_size = images.shape[0]
    rank = vols.shape[0]

    projected_eigenvecs = vol_forward(vols, nufft_plans, filters)

    images = images.reshape((batch_size, -1, 1))
    projected_eigenvecs = projected_eigenvecs.reshape((batch_size, rank, -1))

    projcted_images = torch.matmul(projected_eigenvecs, images)  # size (batch, rank,1)

    m = (
        torch.eye(rank, device=vols.device, dtype=vols.dtype).unsqueeze(0)
        + projected_eigenvecs @ projected_eigenvecs.transpose(1, 2) / noise_var
    )
    mean_m = m.diagonal(dim1=-2, dim2=-1).abs().sum(dim=1) / m.shape[-1]
    projcted_images_transformed = torch.linalg.solve(m / mean_m.reshape(-1, 1, 1), projcted_images) / mean_m.reshape(
        -1, 1, 1
    )  # size (batch, rank,1)
    ml_exp_term = (
        -1
        / (noise_var**2)
        * torch.matmul(projcted_images.transpose(1, 2).conj(), projcted_images_transformed).squeeze()
    )
    ml_noise_term = torch.logdet(m)

    if apply_mean_const_term:
        ml_exp_term += 1 / noise_var * torch.norm(images, dim=(1, 2)) ** 2
        # ml_exp_term += (L**2) * torch.log(noise_var) # constant term

    cost_val = 0.5 * torch.mean(ml_exp_term + ml_noise_term) if mean_aggregate else 0.5 * (ml_exp_term + ml_noise_term)

    if fourier_reg is not None and reg_scale != 0:
        vols_fourier = centered_fft3(vols)
        vols_fourier *= torch.sqrt(fourier_reg / noise_var)
        reg_cost = torch.sum(torch.norm(vols_fourier.reshape((rank, -1)), dim=1) ** 2) / 2
        cost_val += reg_scale * reg_cost

    return cost_val


# TODO : merge this into a single function in cost
def cost_fourier_domain(
    vols,
    images,
    nufft_plans,
    filters,
    noise_var,
    reg_scale=0,
    fourier_reg=None,
    apply_mean_const_term=False,
    mean_aggregate=True,
):
    rank = vols[0].shape[0]
    L = images.shape[-1]
    projected_vols = vol_forward(vols, nufft_plans, filters, fourier_domain=True)

    cost_val = raw_cost_fourier_domain(
        projected_vols, images, noise_var, apply_mean_const_term=apply_mean_const_term, mean_aggregate=mean_aggregate
    )

    if fourier_reg is not None and reg_scale != 0:
        # TODO: vols should get Covar's grid_correction reversed here
        vols_fourier = vols * torch.sqrt(fourier_reg)
        vols_fourier = vols_fourier.reshape((rank, -1))
        vols_fourier_inner_prod = vols_fourier @ vols_fourier.conj().T
        reg_cost = torch.sum(torch.pow(vols_fourier_inner_prod.abs(), 2))
        cost_val += reg_scale * reg_cost

    return cost_val / (L**4)  # Cost value in fourier domain scales with L^4 compared to spatial domain


def raw_cost_fourier_domain(projected_vols, images, noise_var, apply_mean_const_term=False, mean_aggregate=True):
    batch_size = images.shape[0]
    L = images.shape[-1]

    images = images.reshape((batch_size, 1, -1))
    projected_vols = projected_vols.reshape((batch_size, -1, L**2))

    images_projvols_term = torch.matmul(projected_vols, images.transpose(1, 2).conj())
    projvols_prod_term = torch.matmul(projected_vols, projected_vols.transpose(1, 2).conj())

    cost_val = -2 * torch.sum(torch.pow(images_projvols_term.abs(), 2), dim=(1, 2)) + torch.sum(
        torch.pow(projvols_prod_term.abs(), 2), dim=(1, 2)
    )

    # Add noise cost terms
    norm_squared_projvols = torch.diagonal(
        projvols_prod_term, dim1=1, dim2=2
    ).real  # This should be real already but this ensures the dtype gets actually converted
    cost_val += 2 * noise_var * (torch.sum(norm_squared_projvols, dim=1))

    if apply_mean_const_term:
        norm_squared_images = torch.pow(torch.norm(images, dim=(1, 2)), 2)
        cost_val += torch.pow(norm_squared_images, 2)
        cost_val -= 2 * noise_var * norm_squared_images + (noise_var * L) ** 2

    cost_val = torch.mean(cost_val, dim=0) if mean_aggregate else cost_val

    return cost_val


def cost_maximum_liklihood_fourier_domain(
    vols,
    images,
    nufft_plans,
    filters,
    noise_var,
    reg_scale=0,
    fourier_reg=None,
    apply_mean_const_term=False,
    mean_aggregate=True,
):
    rank = vols.shape[0]

    projected_eigenvecs = vol_forward(vols, nufft_plans, filters, fourier_domain=True)

    cost_val = raw_cost_maximum_liklihood_fourier_domain(
        projected_eigenvecs,
        images,
        noise_var,
        apply_mean_const_term=apply_mean_const_term,
        mean_aggregate=mean_aggregate,
    )

    if fourier_reg is not None and reg_scale != 0:
        vols_fourier = vols * torch.sqrt(fourier_reg / noise_var)
        reg_cost = torch.sum(torch.norm(vols_fourier.reshape((rank, -1)), dim=1) ** 2) / 2
        cost_val += reg_scale * reg_cost

    return cost_val


def raw_cost_maximum_liklihood_fourier_domain(
    projected_eigenvecs, images, noise_var, apply_mean_const_term=False, mean_aggregate=True
):

    latent_coords, m, projected_images = compute_latentMAP_batch(images, projected_eigenvecs, noise_var)
    ml_exp_term = -torch.matmul(projected_images.transpose(1, 2).conj(), latent_coords).squeeze()

    ml_noise_term = torch.logdet(m)  # +(L**2) * torch.log(torch.tensor(noise_var)) term which is constant
    if apply_mean_const_term:
        ml_exp_term += 1 / noise_var * torch.norm(images, dim=(1, 2)) ** 2
        # ml_noise_term += (L**2 - rank) * torch.log(noise_var)

    cost_val = (
        0.5 * torch.mean(ml_exp_term + ml_noise_term).real
        if mean_aggregate
        else 0.5 * (ml_exp_term + ml_noise_term).real
    )

    return cost_val


def raw_cost_posterior_maximum_liklihood_fourier_domain(
    projected_eigenvecs, images, noise_var, apply_mean_const_term=False, mean_aggregate=True
):
    latent_coords, m, _ = compute_latentMAP_batch(images, projected_eigenvecs, noise_var)

    projected_eigenvecs = projected_eigenvecs.reshape(m.shape[0], m.shape[-1], -1)
    images = images.reshape(m.shape[0], -1, 1)
    images = images - projected_eigenvecs.transpose(1, 2) @ latent_coords

    latent_coords, m_posterior, projected_images = compute_latentMAP_batch(
        images, projected_eigenvecs, noise_var, eigenvals_inv=m
    )

    ml_exp_term = (
        1 / noise_var * torch.norm(images, dim=(1, 2)) ** 2
        - torch.matmul(projected_images.transpose(1, 2).conj(), latent_coords).squeeze()
    )

    ml_noise_term = torch.logdet(m_posterior) - torch.logdet(
        m
    )  # +(L**2) * torch.log(torch.tensor(noise_var)) term which is constant

    # ml_noise_term += (L**2 - rank) * torch.log(noise_var)

    cost_val = (
        0.5 * torch.mean(ml_exp_term + ml_noise_term).real
        if mean_aggregate
        else 0.5 * (ml_exp_term + ml_noise_term).real
    )

    return cost_val


def frobeniusNorm(vecs):
    # Returns the frobenius norm of a symmetric matrix given by its eigenvectors
    # (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    vecs = vecs.reshape(vecs.shape[0], -1)
    vecs_inn_prod = torch.matmul(vecs, vecs.transpose(0, 1).conj())
    return torch.sqrt(torch.sum(torch.pow(vecs_inn_prod, 2)))


def frobeniusNormDiff(vec1, vec2):
    # returns the frobenius norm of the diffrence of two symmetric matrices given by their eigenvectors
    # (multiplied by the corresponding sqrt(eigenval)) (assuming row vectors as input)
    vec1 = vec1.reshape(vec1.shape[0], -1)
    vec2 = vec2.reshape(vec2.shape[0], -1)

    normdiff_squared = (
        torch.pow(frobeniusNorm(vec1), 2)
        + torch.pow(frobeniusNorm(vec2), 2)
        - 2 * torch.sum(torch.pow(torch.matmul(vec1, vec2.transpose(0, 1).conj()), 2))
    )

    return torch.sqrt(normdiff_squared)


def evalCovarEigs(dataset, eigs, batch_size=8, reg_scale=0, fourier_reg=None):
    device = eigs.device
    num_eigs = eigs.shape[0]
    L = eigs.shape[1]
    nufft_plans = NufftPlan((L,) * 3, batch_size=num_eigs, dtype=eigs.dtype, device=device)
    cost_val = 0
    for i in range(0, len(dataset), batch_size):
        images, pts_rot, filters, _ = dataset[i : i + batch_size]
        pts_rot = pts_rot.to(device)
        images = images.to(device)
        filters = filters.to(device)
        nufft_plans.setpts(pts_rot)
        cost_term = (
            cost(eigs, images, nufft_plans, filters, dataset.noise_var, reg_scale=reg_scale, fourier_reg=fourier_reg)
            * batch_size
        )
        cost_val += cost_term

    return cost_val / len(dataset)


def trainCovar(
    covar_model,
    dataset,
    batch_size,
    optimize_pose=False,
    mean_model=None,
    pose=None,
    savepath=None,
    gt_data=None,
    **kwargs,
):
    num_workers = min(4, get_cpu_count() - 1)
    use_halfsets = kwargs.pop("use_halfsets")
    num_reg_update_iters = kwargs.pop("num_reg_update_iters", None)
    num_epochs = kwargs.pop("max_epochs")
    device = covar_model.device

    if not use_halfsets:
        dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=10,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device=str(device),
        )
        # from torchtnt.utils.data.data_prefetcher import CudaDataPrefetcher
        # dataloader = CudaDataPrefetcher(dataloader,device=device,num_prefetch_batches=4)
        # #TODO : should this be used here? doesn't seem to improve perforamnce
        if not optimize_pose:
            trainer = CovarTrainer(covar_model, dataloader, device, savepath, gt_data=gt_data)
        else:
            trainer = CovarPoseTrainer(covar_model, dataloader, device, mean_model, pose, savepath, gt_data=gt_data)

        trainer.setup_training(**kwargs)
        for _ in range(num_reg_update_iters):
            trainer.train_epochs(num_epochs, restart_optimizer=True)
            eigenvecs = trainer.covar.eigenvecs
            eigenvecs = eigenvecs[0] * (eigenvecs[1] ** 0.5).reshape(-1, 1, 1, 1)
            trainer.compute_fourier_reg_term(eigenvecs)
            trainer.covar.orthogonal_projection()
        trainer.train_epochs(num_epochs, restart_optimizer=True)
        trainer.complete_training()

    else:
        covar_model_copy = copy.deepcopy(covar_model)
        # Reinitalize the copied model since having the same initalization
        # will lead to unwanted correlation even after training.
        with torch.no_grad():
            covar_model_copy.set_vectors(covar_model_copy.init_random_vectors(covar_model.rank))
        half1, half2, permutation = dataset.half_split()

        # TODO: Use sampler like in DDP to not have to split dataset?
        dataloader1 = create_dataloader(
            half1,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=10,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device=str(device),
        )
        dataloader2 = create_dataloader(
            half2,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=10,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device=str(device),
        )

        if not optimize_pose:
            trainer1 = CovarTrainer(covar_model, dataloader1, device, savepath, gt_data=gt_data)
            trainer2 = CovarTrainer(
                covar_model_copy, dataloader2, covar_model_copy.device, save_path=None, gt_data=gt_data
            )
        else:
            mean_model_copy = copy.deepcopy(mean_model)
            pose1, pose2 = pose.split_module(permutation)
            gt_data1, gt_data2 = gt_data.half_split(permutation)
            trainer1 = CovarPoseTrainer(covar_model, dataloader1, device, mean_model, pose1, savepath, gt_data=gt_data1)
            trainer2 = CovarPoseTrainer(
                covar_model_copy,
                dataloader2,
                covar_model_copy.device,
                mean_model_copy,
                pose2,
                save_path=None,
                gt_data=gt_data2,
            )

        trainer1.setup_training(**kwargs)
        trainer2.setup_training(**kwargs)

        for i in range(0, num_reg_update_iters):
            trainer2.to("cpu")
            trainer1.train_epochs(num_epochs, restart_optimizer=True)
            trainer1.to("cpu")
            trainer2.to(device)
            trainer2.train_epochs(num_epochs, restart_optimizer=True)
            trainer1.to(device)
            update_fourier_reg(trainer1, trainer2)

        # trainer1.complete_training()
        trainer2.complete_training()

        del trainer2

        torch.cuda.empty_cache()

        # Train on full dataset
        full_dataloader = create_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=10,
            persistent_workers=True,
            pin_memory=True,
            pin_memory_device=str(device),
        )
        trainer1.train_data = full_dataloader
        if optimize_pose:
            pose.update_from_modules(pose1, pose2, permutation)
            # TODO: merge mean module instead of taking the first one
            trainer1.set_pose_module(pose)
            trainer1.gt_data = gt_data

        trainer1.train(max_epochs=num_epochs, **kwargs)

    torch.cuda.empty_cache()

    return
