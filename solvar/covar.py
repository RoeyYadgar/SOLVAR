from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from solvar.fsc_utils import expand_fourier_shell
from solvar.projection_funcs import centered_fft3, centered_ifft3, crop_tensor


class VolumeBase(torch.nn.Module):
    """Base class for volume operations with Fourier domain support.

    Provides common functionality for volume classes including grid correction
    and domain conversion between spatial and Fourier representations.

    Attributes:
        resolution: Volume resolution (cubic volumes assumed)
        dtype: Data type for volume tensors
        _in_spatial_domain: Whether volume is currently in spatial domain
        upsampling_factor: Factor for Fourier domain upsampling
        grid_correction: Grid correction tensor for NUFFT approximation
    """

    def __init__(
        self,
        resolution: int,
        dtype: torch.dtype = torch.float32,
        fourier_domain: bool = False,
        upsampling_factor: int = 2,
    ) -> None:
        """Initialize VolumeBase.

        Args:
            resolution: Volume resolution (cubic volumes assumed)
            dtype: Data type for volume tensors
            fourier_domain: Whether to start in Fourier domain
            upsampling_factor: Factor for Fourier domain upsampling
        """
        super().__init__()
        self.resolution = resolution
        self.dtype = dtype
        self._in_spatial_domain = not fourier_domain
        self.upsampling_factor = upsampling_factor
        self.grid_correction = None

    def init_grid_correction(self, nufft_disc: str) -> None:
        """Initialize grid correction for NUFFT approximation.

        Args:
            nufft_disc: NUFFT discretization method ("bilinear", "nearest", or other)
        """
        if nufft_disc != "bilinear" and nufft_disc != "nearest":
            self.grid_correction = None
            return

        pixel_pos = torch.arange(-(self.resolution // 2), (self.resolution - 1) // 2 + 1) / self.resolution
        pixel_pos = torch.pi * pixel_pos / self.upsampling_factor
        sinc_val = torch.sin(pixel_pos) / pixel_pos
        sinc_val[pixel_pos == 0] = 1
        sinc_val[sinc_val < 1e-6] = 1

        if nufft_disc == "bilinear":
            sinc_val = sinc_val**2

        sinc_volume = torch.einsum("i,j,k->ijk", sinc_val, sinc_val, sinc_val)
        self.grid_correction = sinc_volume.to(self.device)

    def to(self, *args: Any, **kwargs: Any) -> "VolumeBase":
        """Move module to device and update grid correction.

        Args:
            *args: Positional arguments for torch.nn.Module.to()
            **kwargs: Keyword arguments for torch.nn.Module.to()

        Returns:
            Self for method chaining
        """
        super().to(*args, **kwargs)
        if self.grid_correction is not None:
            self.grid_correction = self.grid_correction.to(*args, **kwargs)
        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """Get state dictionary including domain and grid correction info.

        Args:
            *args: Positional arguments for super().state_dict()
            **kwargs: Keyword arguments for super().state_dict()

        Returns:
            State dictionary with additional volume-specific information
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update(
            {
                "_in_spatial_domain": self._in_spatial_domain,
                "grid_correction": self.grid_correction.to("cpu") if self.grid_correction is not None else None,
            }
        )
        return state_dict

    def load_state_dict(self, state_dict: dict, *args: Any, **kwargs: Any) -> None:
        """Load state dictionary including domain and grid correction info.

        Args:
            state_dict: State dictionary to load
            *args: Positional arguments for super().load_state_dict()
            **kwargs: Keyword arguments for super().load_state_dict()
        """
        self._in_spatial_domain = state_dict.pop("_in_spatial_domain")
        self.grid_correction = state_dict.pop("grid_correction")
        super().load_state_dict(state_dict, *args, **kwargs)
        return


class Mean(VolumeBase):
    """Mean volume module.

    Represents a mean volume by separating it into normalized tensor and its log amplitude.
    This allows for better optimization of the volume magnitude separately from its shape.

    Attributes:
        volume: Normalized volume tensor (parameter)
        log_volume_amplitude: Log of volume amplitude (parameter)
        volume_mask: Optional volume mask for masking
    """

    def __init__(
        self,
        volume_init: torch.Tensor,
        resolution: int,
        dtype: torch.dtype = torch.float32,
        fourier_domain: bool = False,
        volume_mask: Optional[torch.Tensor] = None,
        upsampling_factor: int = 2,
    ) -> None:
        """Initialize Mean volume.

        Args:
            volume_init: Initial volume tensor
            resolution: Volume resolution
            dtype: Data type for tensors
            fourier_domain: Whether to start in Fourier domain
            volume_mask: Optional volume mask
            upsampling_factor: Factor for Fourier domain upsampling
        """
        super().__init__(
            resolution=resolution, dtype=dtype, fourier_domain=fourier_domain, upsampling_factor=upsampling_factor
        )

        volume, log_volume_amplitude = self._get_mean_representation(volume_init.squeeze(0).unsqueeze(0))
        self.volume = torch.nn.Parameter(volume)
        self.log_volume_amplitude = torch.nn.Parameter(log_volume_amplitude)

        self.volume_mask = volume_mask

    def _get_mean_representation(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate volume into normalized tensor and its log amplitude.

        Args:
            volume: Input volume tensor

        Returns:
            Tuple of (normalized_volume, log_amplitude)
        """
        volume_amplitude = volume.reshape(volume.shape[0], -1).norm(dim=1)

        return volume / volume_amplitude, torch.log(volume_amplitude)

    def set_mean(self, volume: torch.Tensor) -> None:
        """Set the mean volume from a tensor.

        Args:
            volume: New volume tensor to set as mean
        """
        volume, log_volume_amplitude = self._get_mean_representation(volume.unsqueeze(0))
        self.volume.data.copy_(volume)
        self.log_volume_amplitude.data.copy_(log_volume_amplitude)

    @property
    def device(self) -> torch.device:
        """Get the device of the volume parameter.

        Returns:
            Device where the volume tensor is located
        """
        return self.volume.device

    def get_volume_fourier_domain(self) -> torch.Tensor:
        """Get volume in Fourier domain.

        Returns:
            Volume tensor in Fourier domain with upsampling
        """
        volume = (
            self.get_volume_spatial_domain() / self.grid_correction
            if self.grid_correction is not None
            else self.get_volume_spatial_domain()
        )
        return centered_fft3(volume, padding_size=(self.resolution * self.upsampling_factor,) * 3)

    def get_volume_spatial_domain(self) -> torch.Tensor:
        """Get volume in spatial domain.

        Returns:
            Volume tensor in spatial domain with amplitude applied
        """
        return self.volume * torch.exp(self.log_volume_amplitude)

    def forward(self, dummy_var: Optional[Any] = None) -> torch.Tensor:
        """Forward pass returning volume in appropriate domain.

        Args:
            dummy_var: Dummy variable for DDP compatibility

        Returns:
            Volume tensor in current domain (spatial or Fourier)
        """
        return self.get_volume_spatial_domain() if self._in_spatial_domain else self.get_volume_fourier_domain()

    def get_volume_mask(self) -> Optional[torch.Tensor]:
        """Get volume mask in appropriate domain.

        Returns:
            Volume mask tensor in current domain, or None if no mask
        """
        if self.volume_mask is None:
            return None
        if not self._in_spatial_domain:
            return centered_fft3(
                self.volume_mask / self.grid_correction if self.grid_correction is not None else self.volume_mask,
                padding_size=(self.resolution * self.upsampling_factor,) * 3,
            )
        else:
            return self.volume_mask

    def to(self, *args: Any, **kwargs: Any) -> "Mean":
        """Move module to device and update volume mask.

        Args:
            *args: Positional arguments for torch.nn.Module.to()
            **kwargs: Keyword arguments for torch.nn.Module.to()

        Returns:
            Self for method chaining
        """
        super().to(*args, **kwargs)
        self.volume_mask = self.volume_mask.to(*args, **kwargs) if self.volume_mask is not None else None

        return self

    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """Get state dictionary including volume mask.

        Args:
            *args: Positional arguments for super().state_dict()
            **kwargs: Keyword arguments for super().state_dict()

        Returns:
            State dictionary with volume mask information
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({"volume_mask": self.volume_mask.to("cpu") if self.volume_mask is not None else None})
        return state_dict

    def load_state_dict(self, state_dict: dict, *args: Any, **kwargs: Any) -> None:
        """Load state dictionary including volume mask.

        Args:
            state_dict: State dictionary to load
            *args: Positional arguments for super().load_state_dict()
            **kwargs: Keyword arguments for super().load_state_dict()
        """
        self.volume_mask = state_dict.pop("volume_mask")
        super().load_state_dict(state_dict, *args, **kwargs)
        return


class Covar(VolumeBase):
    """Covariance matrix representation using low-rank factorization.

    Uses parameterization with normalized eigenvectors and log of square root of eigenvalues.

    Attributes:
        rank: Rank of the covariance matrix
        pixel_var_estimate: Estimate of pixel variance for initialization of eigenvalues
        vectors: Normalized eigenvector parameters
        log_sqrt_eigenvals: Log of square root of eigenvalues
    """

    def __init__(
        self,
        resolution: int,
        rank: int,
        dtype: torch.dtype = torch.float32,
        pixel_var_estimate: Union[float, torch.Tensor] = 1,
        fourier_domain: bool = False,
        upsampling_factor: int = 2,
        vectors: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize Covar with low-rank factorization.

        Args:
            resolution: Volume resolution
            rank: Rank of the covariance matrix
            dtype: Data type for tensors
            pixel_var_estimate: Estimate of pixel variance for initialization
            fourier_domain: Whether to start in Fourier domain
            upsampling_factor: Factor for Fourier domain upsampling
            vectors: Optional initial vectors (if None, random initialization)
        """
        super().__init__(
            resolution=resolution, dtype=dtype, fourier_domain=fourier_domain, upsampling_factor=upsampling_factor
        )
        self.rank = rank
        self.pixel_var_estimate = pixel_var_estimate

        if vectors is None:
            vectors = (
                self.init_random_vectors(rank)
                if (not isinstance(pixel_var_estimate, torch.Tensor) or pixel_var_estimate.ndim == 0)
                else self.init_random_vectors_from_psd(rank, self.pixel_var_estimate)
            )
        else:
            vectors = torch.clone(vectors)

        self._init_parameters(vectors)

    def _init_parameters(self, vectors: torch.Tensor) -> None:
        """Initialize parameters from vectors.

        Args:
            vectors: Initial vector tensor
        """
        vectors, log_sqrt_eigenvals = self._get_eigenvecs_representation(vectors)
        self.vectors = torch.nn.Parameter(vectors)
        self.log_sqrt_eigenvals = torch.nn.Parameter(log_sqrt_eigenvals)

    def _get_eigenvecs_representation(self, vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate vectors into normalized shape and log eigenvalues.

        Args:
            vectors: Input vector tensor

        Returns:
            Tuple of (normalized_vectors, log_sqrt_eigenvals)
        """
        sqrt_eigenvals = vectors.reshape(vectors.shape[0], -1).norm(dim=1)

        return vectors / sqrt_eigenvals.reshape(-1, 1, 1, 1), torch.log(sqrt_eigenvals)

    @property
    def device(self) -> torch.device:
        """Get the device of the vectors parameter.

        Returns:
            Device where the vectors tensor is located
        """
        return self.vectors.device

    def get_vectors(self) -> torch.Tensor:
        """Get vectors in appropriate domain.

        Returns:
            Vector tensor in current domain (spatial or Fourier)
        """
        return self.get_vectors_spatial_domain() if self._in_spatial_domain else self.get_vectors_fourier_domain()

    def set_vectors(self, new_vectors: torch.Tensor) -> None:
        """Set new vectors and update parameters.

        Args:
            new_vectors: New vector tensor to set
        """
        new_vectors, log_sqrt_eigenvals = self._get_eigenvecs_representation(new_vectors)
        self.vectors.data.copy_(new_vectors)
        self.log_sqrt_eigenvals.data.copy_(log_sqrt_eigenvals)

    def init_random_vectors(self, num_vectors: int) -> torch.Tensor:
        """Initialize random vectors.

        Args:
            num_vectors: Number of vectors to generate

        Returns:
            Random vector tensor
        """
        return (torch.randn((num_vectors,) + (self.resolution,) * 3, dtype=self.dtype)) * (self.pixel_var_estimate**0.5)

    def init_random_vectors_from_psd(self, num_vectors: int, psd: torch.Tensor) -> torch.Tensor:
        """Initialize random vectors from power spectral density.

        Args:
            num_vectors: Number of vectors to generate
            psd: Power spectral density tensor

        Returns:
            Random vector tensor with PSD-based statistics
        """
        if psd.ndim == 1:  # If psd input is radial
            psd = expand_fourier_shell(psd, self.resolution, 3)
        vectors = torch.randn((num_vectors,) + (self.resolution,) * 3, dtype=self.dtype)
        vectors_fourier = centered_fft3(vectors) / (self.resolution**1.5)
        vectors_fourier *= torch.sqrt(psd)
        vectors = centered_ifft3(vectors_fourier).real
        return vectors

    def forward(self, dummy_var: Optional[Any] = None) -> torch.Tensor:
        """Forward pass returning vectors in appropriate domain.

        Args:
            dummy_var: Dummy variable for DDP compatibility

        Returns:
            Vector tensor in current domain (spatial or Fourier)
        """
        return self.get_vectors()

    @property
    def eigenvecs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get eigenvectors and eigenvalues from current vectors.

        Returns:
            Tuple of (eigenvectors, eigenvalues)
        """
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().clone().reshape(self.rank, -1)
            _, eigenvals, eigenvecs = torch.linalg.svd(vectors, full_matrices=False)
            eigenvecs = eigenvecs.reshape((self.rank, self.resolution, self.resolution, self.resolution))
            eigenvals = eigenvals**2
            return eigenvecs, eigenvals

    def grad_lr_factor(self) -> List[Dict[str, Union[torch.nn.Parameter, float]]]:
        """Get learning rate factors for different parameters.

        Returns:
            List of parameter groups with learning rates
        """
        return [{"params": self.vectors, "lr": 1}, {"params": self.log_sqrt_eigenvals, "lr": 100}]

    def get_vectors_fourier_domain(self):
        vectors = (
            self.get_vectors_spatial_domain() / self.grid_correction
            if self.grid_correction is not None
            else self.get_vectors_spatial_domain()
        )
        return centered_fft3(vectors, padding_size=(self.resolution * self.upsampling_factor,) * 3)

    def get_vectors_spatial_domain(self):
        return self.vectors * torch.exp(self.log_sqrt_eigenvals).reshape(-1, 1, 1, 1)

    def orthogonal_projection(self):
        with torch.no_grad():
            vectors = self.get_vectors_spatial_domain().reshape(self.rank, -1)
            _, S, V = torch.linalg.svd(vectors, full_matrices=False)
            orthogonal_vectors = (S.reshape(-1, 1) * V).reshape(
                self.rank, self.resolution, self.resolution, self.resolution
            )
            self.set_vectors(orthogonal_vectors)


class CovarFourier(Covar):
    """Used to optimize the covariance eigenvecs in Fourier domain.

    Differs from Covar with `fourier_domain=True` by keeping the underlying vectors in Fourier
    domain directly.
    """

    def __init__(
        self,
        resolution,
        rank,
        dtype=torch.float32,
        pixel_var_estimate=1,
        fourier_domain=True,
        upsampling_factor=2,
        vectors=None,
    ):
        assert fourier_domain, "CovarFourier should always be in Fourier domain."
        super().__init__(
            resolution=resolution,
            rank=rank,
            dtype=dtype,
            pixel_var_estimate=pixel_var_estimate,
            fourier_domain=fourier_domain,
            upsampling_factor=upsampling_factor,
            vectors=vectors,
        )

    def _init_parameters(self, vectors):
        vectors = centered_fft3(vectors, padding_size=(self.resolution * self.upsampling_factor,) * 3)
        vectors, log_sqrt_eigenvals = self._get_eigenvecs_representation(vectors)

        # Params are split into real and imaginary parts since DDP does not support complex params for some reason.
        self._vectors_real = torch.nn.Parameter(vectors.real)
        self._vectors_imag = torch.nn.Parameter(vectors.imag)
        self.log_sqrt_eigenvals = torch.nn.Parameter(log_sqrt_eigenvals)

    def set_vectors(self, vectors):
        if not vectors.is_complex():
            vectors = centered_fft3(vectors, padding_size=(self.resolution * self.upsampling_factor,) * 3)
        vectors, log_sqrt_eigenvals = self._get_eigenvecs_representation(vectors)
        # Store real and imaginary parts separately
        self._vectors_real.data.copy_(vectors.real)
        self._vectors_imag.data.copy_(vectors.imag)
        self.log_sqrt_eigenvals.data.copy_(log_sqrt_eigenvals)

    def get_vectors_spatial_domain(self):
        spatial_vectors = centered_ifft3(self.get_vectors_fourier_domain()).real
        return crop_tensor(spatial_vectors, (self.resolution,) * 3, dims=[-1, -2, -3])

    def get_vectors_fourier_domain(self):
        return torch.complex(self._vectors_real, self._vectors_imag) * torch.exp(self.log_sqrt_eigenvals).reshape(
            -1, 1, 1, 1
        )

    @property
    def device(self):
        return self._vectors_real.device

    def grad_lr_factor(self):
        return [
            {"params": self._vectors_real, "lr": 1},
            {"params": self._vectors_imag, "lr": 1},
            {"params": self.log_sqrt_eigenvals, "lr": 100},
        ]
