from typing import Any, Dict, Tuple

import torch
from torch import distributed as dist

from solvar.covar import Covar
from solvar.covar_sgd import CovarTrainer, cost
from solvar.nufft_plan import NufftPlan
from solvar.wiener_coords import wiener_coords


class IterativeCovarTrainer(CovarTrainer):
    """Trainer for iterative covariance estimation.

    Extends CovarTrainer to perform iterative estimation by training one eigenvector at a time and
    subtracting its contribution from the data before training the next.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the iterative trainer.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)
        self._orig_dataset = self.train_data.dataset

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Train the iterative covariance model.

        Trains one eigenvector at a time, fixing each one and subtracting its
        contribution from the data before training the next.

        Args:
            *args: Positional arguments passed to parent train method
            **kwargs: Keyword arguments passed to parent train method
        """
        for i in range(self.covar.rank):
            super().train(*args, **kwargs)

            self.covar.fix_vector()
            eigenvecs, eigenvals = self.covar.eigenvecs
            if not self.isDDP:
                coords, eigen_forward = wiener_coords(
                    self._orig_dataset, eigenvecs, eigenvals, return_eigen_forward=True
                )
                coords = coords.to("cpu")
                eigen_span_im = torch.sum(coords[:, :, None, None] * eigen_forward, dim=1)
                self.train_data.dataset.images = self._orig_dataset.images - eigen_span_im
            else:
                data_len = len(self.train_data.dataset)
                rank = self.process_ind[0]
                world_size = self.process_ind[1]
                samples_per_process = data_len // world_size
                start_ind = rank * samples_per_process
                end_ind = start_ind + samples_per_process if rank != world_size - 1 else data_len
                coords, eigen_forward = wiener_coords(
                    self._orig_dataset,
                    eigenvecs,
                    eigenvals,
                    start_ind=start_ind,
                    end_ind=end_ind,
                    return_eigen_forward=True,
                )
                coords = coords.to("cpu")
                eigen_span_im = torch.sum(coords[:, :, None, None] * eigen_forward, dim=1)

                updated_parts = [torch.zeros_like(eigen_span_im) for _ in range(world_size)]
                dist.all_gather(updated_parts, eigen_span_im)
                complete_eigen_span_im = torch.cat(updated_parts)
                self.train_data.dataset.images = self._orig_dataset.images - complete_eigen_span_im


class IterativeCovar(Covar):
    """Iterative covariance matrix representation.

    Extends Covar to estimate eigenvectors one at a time, fixing each one before estimating the
    next. This helps with convergence and orthogonality.
    """

    def __init__(
        self, resolution: int, rank: int, dtype: torch.dtype = torch.float32, pixel_var_estimate: float = 1
    ) -> None:
        """Initialize iterative covariance model.

        Args:
            resolution: Volume resolution
            rank: Total rank of the covariance matrix
            dtype: Data type for tensors
            pixel_var_estimate: Estimate of pixel variance for initialization
        """
        super().__init__(resolution, 1, dtype, pixel_var_estimate)
        self.rank = rank

        self.current_estimated_rank = 0

        self.fixed_vectors = torch.zeros((rank, resolution, resolution, resolution), dtype=self.dtype)
        self.fixed_vectors = torch.nn.Parameter(self.fixed_vectors, requires_grad=False)
        self.fixed_vectors_ampl = torch.nn.Parameter(torch.zeros(rank), requires_grad=False)

    @property
    def eigenvecs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get eigenvectors and eigenvalues from fixed vectors.

        Returns:
            Tuple of (eigenvectors, eigenvalues) from currently fixed vectors
        """
        fixed_vectors = self.fixed_vectors[: self.current_estimated_rank].clone().reshape(
            (self.current_estimated_rank, -1)
        ) * self.fixed_vectors_ampl[: self.current_estimated_rank].reshape((self.current_estimated_rank, -1))
        _, eigenvals, eigenvecs = torch.linalg.svd(fixed_vectors, full_matrices=False)
        eigenvecs = eigenvecs.reshape((self.current_estimated_rank, self.resolution, self.resolution, self.resolution))
        eigenvals = eigenvals**2
        return eigenvecs, eigenvals

    def orthogonal_projection(self) -> None:
        """Project current vector onto orthogonal complement of fixed vectors."""
        if self.current_estimated_rank == 0:
            return
        with torch.no_grad():
            vectors = self.vectors.reshape(1, -1)
            fixed_vectors = self.fixed_vectors[: self.current_estimated_rank].reshape((self.current_estimated_rank, -1))
            coeffs = vectors @ fixed_vectors.T
            self.vectors.data.copy_((vectors - coeffs @ fixed_vectors).view_as(self.vectors))

    def fix_vector(self) -> None:
        """Fix the current vector and prepare for next iteration."""
        with torch.no_grad():
            vector = self.vectors.detach().clone()
            vector_ampl = torch.norm(vector).reshape(1)
            normalized_vector = vector / vector_ampl

            self.fixed_vectors.data[self.current_estimated_rank] = normalized_vector
            self.fixed_vectors_ampl.data[self.current_estimated_rank] = vector_ampl
            self.vectors.data.copy_(self.init_random_vectors(1))
            self.current_estimated_rank += 1

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Get state dictionary including current rank.

        Args:
            *args: Positional arguments for super().state_dict()
            **kwargs: Keyword arguments for super().state_dict()

        Returns:
            State dictionary with current estimated rank
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({"current_estimated_rank": self.current_estimated_rank})
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Load state dictionary including current rank.

        Args:
            state_dict: State dictionary to load
            *args: Positional arguments for super().load_state_dict()
            **kwargs: Keyword arguments for super().load_state_dict()
        """
        self.current_estimated_rank = state_dict.pop("current_estimated_rank")
        super().load_state_dict(state_dict, *args, **kwargs)
        return


class IterativeCovarTrainerVer2(CovarTrainer):
    """Alternative iterative trainer with dynamic NUFFT plan updates.

    Updates NUFFT plans as more eigenvectors are estimated to handle the increasing dimensionality
    of the problem.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the alternative iterative trainer.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

    def train(self, *args: Any, **kwargs: Any) -> None:
        """Train with dynamic NUFFT plan updates.

        Args:
            *args: Positional arguments passed to parent train method
            **kwargs: Keyword arguments passed to parent train method
        """
        for i in range(self.covar.rank):
            if i != 0:
                self.nufft_plans = NufftPlan(
                    self.covar.vectors.shape[1:], batch_size=i + 1, dtype=self.covar.vectors.dtype, device=self.device
                )
            super().train(*args, **kwargs)
            self.covar.fix_vector()


class IterativeCovarVer2(IterativeCovar):
    """Alternative iterative covariance with combined cost function.

    Computes cost using both fixed and current vectors together.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the alternative iterative covariance model.

        Args:
            *args: Positional arguments passed to parent class
            **kwargs: Keyword arguments passed to parent class
        """
        super().__init__(*args, **kwargs)

    def cost(
        self,
        images: torch.Tensor,
        nufft_plans: Any,
        filters: torch.Tensor,
        noise_var: float,
        reg: float = 0,
    ) -> torch.Tensor:
        """Compute cost using both fixed and current vectors.

        Args:
            images: Input images
            nufft_plans: NUFFT plans for projection
            filters: CTF filters
            noise_var: Noise variance
            reg: Regularization parameter

        Returns:
            Cost value
        """
        vectors = torch.cat((self.fixed_vectors[: self.current_estimated_rank], self.vectors), dim=0)
        return cost(vectors, images, nufft_plans, filters, noise_var, reg)
