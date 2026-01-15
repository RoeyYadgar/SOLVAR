import inspect
import logging
import math
from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple, Union

import numpy as np
import torch
from aspire.utils import grid_3d
from cufinufft import Plan as cuPlan
from finufft import Plan

logger = logging.getLogger(__name__)


def get_half_fourier_grid(points: torch.Tensor, dim: Tuple[int, int]) -> torch.Tensor:
    """Converts a 2d rotated grid of points in fourier space into half a grid that represents only
    positive frequencies in the first dimension. This is used to reduce number of points to compute
    in  NUFFT since real signals have conjugate symmetry in fourier space.

    Args:
        points: Batch of 2d rotated grids of size (a,...,b,L,L,c,...,d)
        dim: Two tuple of dimensions of the actual grid (For example for shape (a,b,c,L,L) dim=(3,4))

    Returns:
        Batch of 2d rotated half-grids of size (a,...,b,L/2,L,c,...,d)
    """
    L = points.shape[dim[0]]
    indices = [slice(None)] * points.ndim
    indices[dim[0]] = slice(L // 2, L)
    # In even image size fourier points in index 0 corrspond to the 'most negative' frequency
    # which doesnt have a positive counterpart
    if L % 2 == 0:
        indices[dim[1]] = slice(1, L)
    return points[tuple(indices)]


class BaseNufftPlan(ABC):
    """Abstract base class for NUFFT plans.

    Defines the interface for NUFFT (Non-Uniform Fast Fourier Transform) plans used for volume
    projections.
    """

    def setpts(self, points: torch.Tensor) -> None:
        """Set the points for NUFFT computation.

        Args:
            points: Point coordinates for NUFFT
        """
        raise NotImplementedError

    def execute_forward(self, volume: torch.Tensor) -> torch.Tensor:
        """Execute forward NUFFT.

        Args:
            volume: Input volume

        Returns:
            Forward NUFFT result
        """
        raise NotImplementedError


class NufftPlanDiscretized(BaseNufftPlan):
    """Discretized version of the NUFFT operator.

    Specifically designed for volume projection operators rather than general NUFFT usage.
    Uses PyTorch's `grid_sample` function to interpolate from the Fourier transform of the given volume.
    Assumes input volume is already given in frequency domain.

    Attributes:
        sz: Size of the volume
        upsample_factor: Upsampling factor for interpolation
        mode: Interpolation mode for grid_sample
        use_half_grid: Whether to use half-grid optimization
        points_L: Resolution of the point grid
        points: Transformed point coordinates
        batch_points: Number of batch points
    """

    def __init__(
        self, sz: Tuple[int, ...], upsample_factor: int = 1, mode: str = "bilinear", use_half_grid: bool = False
    ) -> None:
        # TODO: fix issue with use_half_grid and cropped nufft points
        assert not use_half_grid, "half_grid not supported with NufftPlanDiscretized"
        self.sz = sz
        self.upsample_factor = upsample_factor
        self.mode = mode
        self.use_half_grid = use_half_grid

    def setpts(self, points: torch.Tensor) -> None:
        """Set points for discretized NUFFT computation.

        Args:
            points: Point coordinates of shape (N, 3, L**2)
        """
        L = self.sz[0]

        # The size points could be different if we want to compute a cropped nufft points
        self.points_L = int(points.shape[-1] ** 0.5)
        self.points = points.transpose(-2, -1).reshape(1, -1, self.points_L, self.points_L, 3)
        self.points = self.points.flip(
            -1
        )  # grid_sample uses xyz convention unlike the zyx given by aspire's `rotated_grids`
        self.points = get_half_fourier_grid(self.points, (2, 3)) if self.use_half_grid else self.points

        vol_L = L * self.upsample_factor
        # For even image sizes fourier points are [-L/2, ... , L/2-1]/(L/2)*pi
        # while torch grid_sample treats grid as [-1 , ... , 1]
        # For odd image sizes fourier points are [-(L-1)/2,...,(L-1)/2]/(L/2)*pi
        self.points *= (vol_L / (vol_L - 1)) / torch.pi
        if vol_L % 2 == 0:
            self.points = self.points + 1 / (vol_L - 1)

        self.batch_points = self.points.shape[1]

    def execute_forward(self, volume: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Execute forward NUFFT using grid sampling.

        Args:
            volume: Volume in Fourier domain with shape (N,L,L,L) either as complex tensor
                   or as tuple pair of real and imaginary tensors

        Returns:
            Forward NUFFT result
        """
        L = self.points_L
        # For some reason grid_sample does not support complex data. Instead the real andimaginary
        # parts are splitted into different 'channels'
        if isinstance(volume, tuple):
            volume_real = volume[0].unsqueeze(1)
            volume_imag = volume[1].unsqueeze(1)
        else:
            volume = volume.unsqueeze(1)
            volume_real = volume.real
            volume_imag = volume.imag
        volume_L = volume_real.shape[
            -1
        ]  # Size of volume can be different than the L since might have been upsampled (or downsampled)
        volume_real_imag_split = torch.cat((volume_real, volume_imag), dim=1).reshape(
            -1, volume_L, volume_L, volume_L
        )  # Shape of (N*2,volume_L,volume_L,volume_L)
        # Grid sample's batch is used when we need to sample different volumes with different grids,
        # here however we want to sample all volumes with different grids so we use the grid_sample channels instead.
        output = torch.nn.functional.grid_sample(
            input=volume_real_imag_split.unsqueeze(0), grid=self.points, mode=self.mode, align_corners=True
        )  # Shape of (1,N*2,n,L,L) (or (1,N*2,n,L/2,L) if self.use_half_grid=True)

        half_L = math.ceil(L / 2) if self.use_half_grid else L
        full_L = (L - 1) if (self.use_half_grid and L % 2 == 0) else L
        # Put it back into its complex form
        output = output.reshape(-1, 2, self.batch_points, half_L, full_L)
        output = torch.complex(output[:, 0], output[:, 1])  # Shape of (N,n,L,L) (or (N,n,L/2,L))

        if (
            self.use_half_grid
        ):  # Fill the full grid by flipping & conjugating the output and concateting with original output
            output_full = torch.concat((output[:, :, 1:, :].flip(dims=(2, 3)).conj(), output), dim=2)
            if L % 2 == 0:  # When image size is even we need to pad the image with additional row and col of zeros
                zeros_row = torch.zeros((output.shape[:2] + (1, L - 1)), dtype=output.dtype, device=output.device)
                output_full = torch.concat((zeros_row, output_full), dim=2)
                zeros_col = torch.zeros((output.shape[:2] + (L, 1)), dtype=output.dtype, device=output.device)
                output_full = torch.concat((zeros_col, output_full), dim=3)

            output = output_full
        return output

    def execute_adjoint(self, signal):
        """Adjoint of the NUFFT projection operator.

        Takes sampled Fourier values and maps them back to a 3D Fourier volume.
        Output shape: (N, L, L, L)
        """
        assert not self.use_half_grid, "half_grid not supported with adjoint operator"

        L = self.sz[0]
        volume_L = L * self.upsample_factor

        # Split complex signal into real and imag channels
        signal_real = signal.real
        signal_imag = signal.imag
        signal_cat = torch.cat(
            (signal_real.unsqueeze(0), signal_imag.unsqueeze(0)), dim=0
        )  # shape: (2,n, L, L) or (or (2,n,L/2,L) if self.use_half_grid=True)
        signal_cat = signal_cat.unsqueeze(0)

        dummy_input = torch.ones(
            1, 2, volume_L, volume_L, volume_L, dtype=signal_cat.dtype, device=signal.device, requires_grad=True
        )
        dummy_output = torch.nn.functional.grid_sample(
            input=dummy_input, grid=self.points, mode=self.mode, align_corners=True
        )

        adjoint_signal = torch.autograd.grad(
            outputs=dummy_output, inputs=dummy_input, grad_outputs=signal_cat, retain_graph=False, create_graph=False
        )[0]
        return torch.complex(adjoint_signal[:, 0], adjoint_signal[:, 1])

    def execute_adjoint_unaggregated(self, signal):
        """Adjoint of the NUFFT projection operator.

        Takes sampled Fourier values and maps them back to a 3D Fourier volume.
        Output shape: (n, L, L, L)
        """
        assert not self.use_half_grid, "half_grid not supported with adjoint operator"

        L = self.sz[0]
        volume_L = L * self.upsample_factor

        # Split complex signal into real and imag channels
        signal_real = signal.real
        signal_imag = signal.imag
        signal_cat = torch.cat(
            (signal_real.unsqueeze(0), signal_imag.unsqueeze(0)), dim=0
        )  # shape: (2,n, L, L) or (or (2,n,L/2,L) if self.use_half_grid=True)
        signal_cat = signal_cat.unsqueeze(0)

        dummy_input = torch.ones(
            signal.shape[0],
            2,
            volume_L,
            volume_L,
            volume_L,
            dtype=signal_cat.dtype,
            device=signal.device,
            requires_grad=True,
        )
        dummy_output = torch.nn.functional.grid_sample(
            input=dummy_input, grid=self.points.transpose(0, 1), mode=self.mode, align_corners=True
        )

        adjoint_signal = torch.autograd.grad(
            outputs=dummy_output,
            inputs=dummy_input,
            grad_outputs=signal_cat.transpose(0, 2),
            retain_graph=False,
            create_graph=False,
        )[0]
        return torch.complex(adjoint_signal[:, 0], adjoint_signal[:, 1])


class NufftPlan(BaseNufftPlan):
    """NUFFT plan using FINUFFT/cuFINUFFT libraries.

    Provides CPU and GPU implementations of non-uniform FFT for volume projections.

    Attributes:
        sz: Size of the volume
        batch_size: Batch size for computation
        device: Device for computation (CPU or GPU)
        dtype: Data type for real numbers
        complex_dtype: Data type for complex numbers
        forward_plan: FINUFFT plan for forward transform
        adjoint_plan: FINUFFT plan for adjoint transform
        points_N: Number of point sets
        points_L: Resolution of point grid
        points: Point coordinates
    """

    def __init__(
        self,
        sz: Tuple[int, ...],
        batch_size: int = 1,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
        **kwargs: Any,
    ) -> None:
        self.sz = sz
        self.batch_size = batch_size
        self.device = device
        if dtype == torch.float32 or dtype == torch.complex64:
            self.dtype = torch.float32
            self.complex_dtype = torch.complex64
            np_dtype = np.float32
            np_complex_dtype = np.complex64
        elif dtype == torch.float64 or dtype == torch.complex128:
            self.dtype = torch.float64
            self.complex_dtype = torch.complex128
            np_dtype = np.float64
            np_complex_dtype = np.complex128

        eps = max(eps, np.finfo(np_dtype).eps * 10)  # dtype determines determines the eps bottleneck

        if device == torch.device("cpu"):
            self.forward_plan = Plan(
                nufft_type=2, n_modes_or_dim=self.sz, n_trans=batch_size, eps=eps, dtype=np_complex_dtype, **kwargs
            )
            self.adjoint_plan = Plan(
                nufft_type=1, n_modes_or_dim=self.sz, n_trans=batch_size, eps=eps, dtype=np_complex_dtype, **kwargs
            )
        else:  # GPU device
            # TODO : check gpu_sort effect
            default_kwargs = {"gpu_method": 1, "gpu_sort": 0}
            default_kwargs.update(kwargs)
            self.forward_plan = cuPlan(
                nufft_type=2,
                n_modes=self.sz,
                n_trans=batch_size,
                eps=eps,
                dtype=np_complex_dtype,
                gpu_device_id=device.index,
                **default_kwargs,
            )
            self.adjoint_plan = cuPlan(
                nufft_type=1,
                n_modes=self.sz,
                n_trans=batch_size,
                eps=eps,
                dtype=np_complex_dtype,
                gpu_device_id=device.index,
                **default_kwargs,
            )

    def setpts(self, points: torch.Tensor) -> None:
        """Set points for NUFFT computation.

        Args:
            points: Point coordinates of shape (N, 3, L**2)
        """
        self.points_N = points.shape[0]
        self.points_L = int(points.shape[-1] ** 0.5)
        points = points.transpose(0, 1).reshape((3, -1))
        points = (torch.remainder(points + torch.pi, 2 * torch.pi) - torch.pi).contiguous()
        self.points = points
        if self.device == torch.device("cpu"):
            points = points.numpy()  # finufft plan does not accept torch tensors
        # Clean references to past points (preventing a memory leak)
        self.forward_plan._references = []
        self.adjoint_plan._references = []

        self.forward_plan.setpts(*points)
        self.adjoint_plan.setpts(*points)

    def execute_forward(self, signal: torch.Tensor) -> torch.Tensor:
        """Execute forward NUFFT.

        Args:
            signal: Input signal tensor

        Returns:
            Forward NUFFT result
        """
        zero_pad = False
        if signal.shape[0] < self.batch_size:
            logger.warning("Signal batch size is smaller than the nufft plan batch size. Padding with zeros")
            zero_pad = True
            orig_batch = signal.shape[0]
            pad_size = self.batch_size - signal.shape[0]
            pad = torch.zeros((pad_size,) + signal.shape[1:], dtype=signal.dtype, device=signal.device)
            signal = torch.cat((signal, pad), dim=0)

        signal = signal.type(self.complex_dtype).contiguous()
        if self.device == torch.device("cpu"):
            signal = signal.numpy()
        forward_signal = self.forward_plan.execute(signal).reshape((self.batch_size, -1))
        if zero_pad:
            forward_signal = forward_signal[:orig_batch]
        if self.device == torch.device("cpu"):
            forward_signal = torch.from_numpy(forward_signal)
        return forward_signal.reshape((self.batch_size, self.points_N, self.points_L, self.points_L))

    def execute_adjoint(self, signal: torch.Tensor) -> torch.Tensor:
        """Execute adjoint NUFFT.

        Args:
            signal: Input signal tensor

        Returns:
            Adjoint NUFFT result
        """
        zero_pad = False
        if signal.shape[0] < self.batch_size:
            logger.warning("Signal batch size is smaller than the nufft plan batch size. Padding with zeros")
            zero_pad = True
            orig_batch = signal.shape[0]
            pad_size = self.batch_size - signal.shape[0]
            pad = torch.zeros((pad_size,) + signal.shape[1:], dtype=signal.dtype, device=signal.device)
            signal = torch.cat((signal, pad), dim=0)

        signal = signal.type(self.complex_dtype).contiguous()
        if self.device == torch.device("cpu"):
            signal = signal.numpy()
        adjoint_signal = self.adjoint_plan.execute(signal.reshape((self.batch_size, -1)))
        if zero_pad:
            adjoint_signal = adjoint_signal[:orig_batch]
        if self.device == torch.device("cpu"):
            adjoint_signal = torch.from_numpy(adjoint_signal)
        return adjoint_signal.reshape((self.batch_size, *self.sz))


class NufftType(Enum):
    DISCRETIZED = NufftPlanDiscretized
    FINUFFT = NufftPlan


@dataclass
class NufftSpec:
    nufft_type: NufftType
    sz: Tuple[int, int, int] = None
    upsample_factor: int = None
    mode: str = None
    batch_size: int = None
    eps: float = None
    dtype: torch.dtype = None
    device: str = None

    def get_nufft_kwargs(self) -> dict:
        """Returns a dictionary of kwargs for constructor the NUFFT type based on the provided
        spec."""

        constructor_sig = inspect.signature(self.nufft_type.__init__)
        valid_params = constructor_sig.parameters.keys()

        return {k: v for k, v in self.__dict__.items() if k in valid_params and v is not None}

    @property
    def input_in_fourier(self) -> bool:
        return self.nufft_type == NufftPlanDiscretized

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise AttributeError(f"{k} is not a valid field")
            setattr(self, k, v)


class TorchNufftForward(torch.autograd.Function):
    """PyTorch autograd function for forward NUFFT with gradient support."""

    @staticmethod
    def forward(
        ctx: Any, signal: torch.Tensor, points: torch.Tensor, nufft_plan: Any, reset_pts: bool = False
    ) -> torch.Tensor:
        if reset_pts:
            nufft_plan.setpts(points)
        # Otherwise assumes nufft_plan.setpts was already called prior
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        ctx.complex_input = signal.is_complex()
        ctx.signal = signal
        return nufft_plan.execute_forward(signal)

    @staticmethod
    def backward(ctx, grad_output):
        # Since nufft_plan referenced is saved on the context, set_pts cannot be called before using
        # backward (otherwise it will compute the wrong adjoint transformation)
        nufft_plan = ctx.nufft_plan
        if ctx.needs_input_grad[0]:
            signal_grad = nufft_plan.execute_adjoint(grad_output).reshape(ctx.signal_shape)
            if not ctx.complex_input:  # If input to forward method is real the gradient should also be real
                signal_grad = signal_grad.real
        else:
            signal_grad = None
        if ctx.needs_input_grad[1]:
            coords = -1j * torch.tensor(
                np.concatenate([grid_3d(15, normalized=False)[v][None, :] for v in ["z", "y", "x"]]),
                dtype=ctx.signal.dtype,
                device=ctx.signal.device,
            )
            points_grad = torch.zeros_like(ctx.nufft_plan.points)
            for k in range(3):
                points_grad[k] = (nufft_plan.execute_forward(ctx.signal * coords[k]).conj() * grad_output).real.sum(
                    dim=0
                )
        else:
            points_grad = None
        return signal_grad, points_grad, None, None


class TorchNufftAdjoint(torch.autograd.Function):
    """PyTorch autograd function for adjoint NUFFT with gradient support."""

    @staticmethod
    def forward(ctx: Any, signal: torch.Tensor, nufft_plan: Any) -> torch.Tensor:
        ctx.signal_shape = signal.shape
        ctx.nufft_plan = nufft_plan
        ctx.complex_input = signal.is_complex()
        return nufft_plan.execute_adjoint(signal)

    @staticmethod
    def backward(ctx, grad_output):
        nufft_plan = ctx.nufft_plan
        signal_grad = nufft_plan.execute_forward(grad_output).reshape(ctx.signal_shape)
        if not ctx.complex_input:  # If input to forward method is real the gradient should also be real
            signal_grad = signal_grad.real
        return signal_grad, None


def nufft_forward(signal: torch.Tensor, nufft_plan: BaseNufftPlan) -> torch.Tensor:
    """Execute forward NUFFT with autograd support.

    Args:
        signal: Input signal tensor
        nufft_plan: NUFFT plan instance

    Returns:
        Forward NUFFT result
    """
    return (
        TorchNufftForward.apply(signal, nufft_plan.points, nufft_plan)
        if isinstance(nufft_plan, NufftPlan)
        else nufft_plan.execute_forward(signal)
    )  # NufftPlan.execute_forward does not have autograd and so we must pass it into the autograd class


def nufft_adjoint(signal: torch.Tensor, nufft_plan: BaseNufftPlan) -> torch.Tensor:
    """Execute adjoint NUFFT with autograd support.

    Args:
        signal: Input signal tensor
        nufft_plan: NUFFT plan instance

    Returns:
        Adjoint NUFFT result
    """
    return (
        TorchNufftAdjoint.apply(signal, nufft_plan)
        if isinstance(nufft_plan, NufftPlan)
        else nufft_plan.execute_adjoint(signal)
    )
