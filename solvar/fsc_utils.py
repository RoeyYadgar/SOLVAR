from typing import Optional, Tuple, Union

import torch
from aspire.utils import grid_2d, grid_3d

from solvar.projection_funcs import centered_fft3


class FourierShell:
    """Class for computing Fourier shell operations on 2D and 3D signals.

    Provides functionality for radial averaging, power spectral density computation,
    and Fourier shell correlation calculations.

    Attributes:
        L: Signal size (cubic signals assumed)
        dim: Signal dimensionality (2 or 3)
        dtype: Data type for computations
        device: Device for computations
        grid_radius: Radial grid values
        radial_values: Radial indices
        shell_size: Number of points in each radial shell
        shell_inds: Indices for each radial shell
    """

    def __init__(
        self, L: int, dim: int, dtype: torch.dtype = torch.float32, device: torch.device = torch.device("cpu")
    ) -> None:
        """Initialize FourierShell.

        Args:
            L: Signal size (cubic signals assumed)
            dim: Signal dimensionality (2 or 3)
            dtype: Data type for computations (default: torch.float32)
            device: Device for computations (default: CPU)
        """
        self.L = L
        self.dim = dim
        self.dtype = dtype
        self.device = device
        if dim == 2:
            grid_func = grid_2d
        elif dim == 3:
            grid_func = grid_3d
        self.grid_radius = torch.tensor(grid_func(L, shifted=False, normalized=False)["r"], dtype=dtype, device=device)
        self.radial_values = torch.arange(0, int(torch.ceil(torch.max(self.grid_radius)).item()))
        self.shell_size, self.shell_inds, self.shell_assignment = self.compute_shell_size()

    @staticmethod
    def from_tensor(tensor: torch.Tensor) -> "FourierShell":
        """Create FourierShell from tensor.

        Args:
            tensor: Input tensor to infer parameters from

        Returns:
            FourierShell instance
        """
        L = tensor.shape[0]
        dim = tensor.ndim
        return FourierShell(L, dim, tensor.dtype, tensor.device)

    def compute_shell_size(self) -> Tuple[torch.Tensor, list, torch.Tensor]:
        """Compute shell sizes, indices, and flat shell assignment for radial averaging.

        Returns:
            Tuple of (shell_sizes, shell_indices, shell_assignment)
            shell_assignment: flat long tensor of shape (L^dim,) mapping each voxel to its
            shell index.  Every voxel is assigned (intervals are half-open [lo, hi)).
        """
        S = len(self.radial_values)
        N = self.grid_radius.numel()
        shell_size = torch.zeros(S, dtype=self.dtype, device=self.device)
        shell_inds = []
        shell_assignment = torch.zeros(N, dtype=torch.long, device=self.device)
        for i in range(S):
            lower_rad_threshold, upper_rad_threshold = self.radial_avg_interval(i)
            shell_ind = (self.grid_radius >= lower_rad_threshold) & (self.grid_radius < upper_rad_threshold)
            shell_size[i] = torch.sum(shell_ind)
            shell_inds.append(torch.where(shell_ind))
            shell_assignment[shell_ind.reshape(-1)] = i
        return shell_size, shell_inds, shell_assignment

    def radial_avg_interval(self, radial_index: int) -> Tuple[float, float]:
        """Get radial averaging interval for given index.

        Args:
            radial_index: Index of radial shell

        Returns:
            Tuple of (lower_threshold, upper_threshold)
        """
        lower_rad_threshold = self.radial_values[radial_index] - 0.5
        upper_rad_threshold = (
            self.radial_values[radial_index] + 0.5
            if (radial_index < len(self.radial_values) - 1)
            else self.radial_values[radial_index] + 1
        )

        return lower_rad_threshold, upper_rad_threshold

    def _unsqueeze_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Add batch dimension if necessary.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with batch dimension

        Raises:
            Exception: If tensor has unexpected number of dimensions
        """
        # Add batch dimension if necessery
        if tensor.ndim == self.dim + 1:
            return tensor
        elif tensor.ndim == self.dim:
            return tensor.unsqueeze(0)
        else:
            raise Exception(f"Tensor dimension should be either {self.dim} or {self.dim+1} (for batch computation)")

    def avergage_fourier_shell(self, spectrum_signals: torch.Tensor) -> torch.Tensor:
        """Compute radial average of Fourier spectrum.

        Args:
            spectrum_signals: Fourier spectrum signals

        Returns:
            Radially averaged spectrum
        """
        spectrum_signals = self._unsqueeze_batch_dim(spectrum_signals)
        n = spectrum_signals.shape[0]
        S = len(self.radial_values)
        flat = spectrum_signals.reshape(n, -1)

        idx = self.shell_assignment.unsqueeze(0).expand(n, -1)
        shell_avg = torch.zeros(n, S, dtype=spectrum_signals.dtype, device=self.device)
        shell_avg.scatter_add_(1, idx, flat)
        shell_avg /= self.shell_size.unsqueeze(0)

        if n == 1:
            return shell_avg[0]

        return shell_avg

    def varaince_fourier_shell(self, spectrum_signals: torch.Tensor) -> torch.Tensor:
        spectrum_signals = self._unsqueeze_batch_dim(spectrum_signals)
        n = spectrum_signals.shape[0]

        shell_var = torch.zeros(n, len(self.radial_values), dtype=spectrum_signals.dtype, device=self.device)
        for i in range(shell_var.shape[1]):
            shell_var[:, i] = torch.var(spectrum_signals[:, *self.shell_inds[i]], dim=-1)

        if n == 1:
            return shell_var[0]

    def rpsd(self, signals: torch.Tensor) -> torch.Tensor:
        """Compute radial power spectral density.

        Args:
            signals: Input signals

        Returns:
            Radial power spectral density
        """
        signals_psd = torch.abs(centered_fft3(self._unsqueeze_batch_dim(signals))) ** 2
        return self.avergage_fourier_shell(signals_psd)

    def sum_over_shell(self, shells: torch.Tensor) -> torch.Tensor:
        """Sum values over radial shells.

        Args:
            shells: Shell values

        Returns:
            Sum over shells
        """
        shells = shells.unsqueeze(0) if shells.ndim == 1 else shells
        n = len(shells)
        shell_sum = torch.sum(shells * self.shell_size, dim=-1)
        if n == 1:
            return shell_sum[0]

        return shell_sum

    def expand_fourier_shell(self, shells: torch.Tensor) -> torch.Tensor:
        """Expand radial shell values to full Fourier space.

        Args:
            shells: Radial shell values

        Returns:
            Expanded Fourier signal
        """
        shells = shells.unsqueeze(0) if shells.ndim == 1 else shells
        n = len(shells)
        fourier_signal = torch.zeros((len(shells),) + (self.L,) * self.dim, dtype=self.dtype, device=self.device)
        for i in range(len(shells[0])):
            fourier_signal[:, *self.shell_inds[i]] = shells[:, i].unsqueeze(1)

        if n == 1:
            return fourier_signal[0]

        return fourier_signal


def concat_tensor_tuple(tensor_tuple: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Concatenate tensor tuple into single tensor.

    Args:
        tensor_tuple: Tuple of tensors to concatenate

    Returns:
        Concatenated tensor
    """
    return torch.cat([t.unsqueeze(0) for t in tensor_tuple])


def average_fourier_shell(*spectrum_signals: torch.Tensor) -> torch.Tensor:
    """Compute average Fourier shell for multiple spectrum signals.

    Args:
        *spectrum_signals: Variable number of spectrum signals

    Returns:
        Averaged Fourier shell
    """
    return FourierShell.from_tensor(spectrum_signals[0]).avergage_fourier_shell(concat_tensor_tuple(spectrum_signals))


def rpsd(*signals: torch.Tensor) -> torch.Tensor:
    """Compute radial power spectral density for multiple signals.

    Args:
        *signals: Variable number of signals

    Returns:
        Radial power spectral density
    """
    return FourierShell.from_tensor(signals[0]).rpsd(concat_tensor_tuple(signals))


def expand_fourier_shell(shells: torch.Tensor, L: int, dim: int) -> torch.Tensor:
    """Expand radial shell values to full tensor in Fourier space.

    Args:
        shells: Radial shell values
        L: Signal size
        dim: Signal dimensionality

    Returns:
        Expanded Fourier signal
    """
    return FourierShell(L, dim, shells.dtype, shells.device).expand_fourier_shell(shells)


def upsample_and_expand_fourier_shell(shells: torch.Tensor, L: int, dim: int) -> torch.Tensor:
    """Upsample and expand radial shell values to full tensor in Fourier space.

    Args:
        shells: Radial shell values
        L: Target signal size
        dim: Signal dimensionality

    Returns:
        Upsampled and expanded Fourier signal
    """
    fourier_shell = FourierShell(L, dim, shells.dtype, shells.device)
    fourier_shells_upsampled = torch.nn.functional.interpolate(
        shells.unsqueeze(0).unsqueeze(0) if shells.ndim == 1 else shells.unsqueeze(0),
        size=len(fourier_shell.shell_size),
        mode="linear",
        align_corners=False,
    ).squeeze()
    return fourier_shell.expand_fourier_shell(fourier_shells_upsampled)


def sum_over_shell(shell: torch.Tensor, L: int, dim: int) -> torch.Tensor:
    """Sum values over radial shells.

    Args:
        shell: Shell values
        L: Signal size
        dim: Signal dimensionality

    Returns:
        Sum over shells
    """
    return FourierShell(L, dim, shell.dtype, shell.device).sum_over_shell(shell)


def vol_fsc(signal1: torch.Tensor, signal2: torch.Tensor) -> torch.Tensor:
    """Compute Fourier shell correlation between two signals.

    Args:
        signal1: First signal
        signal2: Second signal

    Returns:
        Fourier shell correlation values
    """
    signal1_fft = centered_fft3(signal1)
    signal2_fft = centered_fft3(signal2)

    correlation, rpsd1, rpsd2 = average_fourier_shell(
        torch.real(signal1_fft * torch.conj(signal2_fft)), torch.abs(signal1_fft) ** 2, torch.abs(signal2_fft) ** 2
    )

    fsc = correlation / torch.sqrt(rpsd1 * rpsd2)

    return fsc


def covar_correlate(
    fourier_eigenvecs1: torch.Tensor,
    fourier_eigenvecs2: torch.Tensor,
    return_shell_size: bool = False,
    fourier_shell: Optional["FourierShell"] = None,
    chunked: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute correlation matrix between two sets of Fourier-domain eigenvectors.

    Set ``chunked=True`` to trade speed for memory when r or L is large: loops

    Args:
        fourier_eigenvecs1: First set of Fourier eigenvectors, shape (r1, L, L, L)
        fourier_eigenvecs2: Second set of Fourier eigenvectors, shape (r2, L, L, L)
        return_shell_size: Whether to also return shell sizes (default: False)
        fourier_shell: Pre-built FourierShell; created from eigenvecs1 if None
        chunked: Use memory-efficient loop over r1 instead of full vectorisation

    Returns:
        Correlation matrix (S, S), or tuple of (correlation_matrix, shell_sizes)
    """
    L = fourier_eigenvecs1.shape[-1]
    dtype = torch.float32 if fourier_eigenvecs1.dtype == torch.complex64 else torch.float64
    if fourier_shell is None:
        fourier_shell = FourierShell(L, 3, dtype, fourier_eigenvecs1.device)

    r1, r2 = fourier_eigenvecs1.shape[0], fourier_eigenvecs2.shape[0]
    S = len(fourier_shell.radial_values)
    shell_size = fourier_shell.shell_size
    V1 = fourier_eigenvecs1.reshape(r1, -1)  # (r1, L^3)
    V2 = fourier_eigenvecs2.reshape(r2, -1)  # (r2, L^3)
    idx = fourier_shell.shell_assignment.unsqueeze(0)

    if not chunked:
        # All r1·r2 cross-products in one shot: peak memory O(r1·r2·L^3).
        cross = (V1.unsqueeze(1) * V2.unsqueeze(0).conj()).reshape(r1 * r2, -1)
        A = torch.zeros(r1 * r2, S, dtype=fourier_eigenvecs1.dtype, device=fourier_eigenvecs1.device)
        A.scatter_add_(1, idx.expand(r1 * r2, -1), cross)
        A /= shell_size.unsqueeze(0)
        correlate_matrix = A.T @ A.conj()
    else:
        # Loop over r1: peak memory O(r2·L^3).
        correlate_matrix = torch.zeros(S, S, dtype=fourier_eigenvecs1.dtype, device=fourier_eigenvecs1.device)
        idx_r2 = idx.expand(r2, -1)
        for i in range(r1):
            cross_i = V1[i].unsqueeze(0) * V2.conj()  # (r2, L^3)
            A_i = torch.zeros(r2, S, dtype=fourier_eigenvecs1.dtype, device=fourier_eigenvecs1.device)
            A_i.scatter_add_(1, idx_r2, cross_i)
            A_i /= shell_size.unsqueeze(0)
            correlate_matrix += A_i.T @ A_i.conj()

    if return_shell_size:
        return correlate_matrix, shell_size
    return correlate_matrix


def covar_rpsd(
    eigenvectors: torch.Tensor,
    from_fourier: bool = False,
    fourier_shell: Optional["FourierShell"] = None,
    chunked: bool = False,
) -> torch.Tensor:
    """Compute radial power spectral density for covariance eigenvectors.

    Set ``chunked=True`` for large r or L to reduce peak memory from O(r²·N) to O(r·N).

    Args:
        eigenvectors: Eigenvector tensors, shape (r, L, L, L)
        from_fourier: Whether input is already in Fourier domain (default: False)
        fourier_shell: Pre-built FourierShell; created from eigenvectors if None
        chunked: Forward to covar_correlate's chunked mode (default: False)

    Returns:
        Radial power spectral density matrix, shape (S, S)
    """
    V = eigenvectors if from_fourier else centered_fft3(eigenvectors)
    L = V.shape[-1]
    dtype = torch.float32 if V.dtype == torch.complex64 else torch.float64
    if fourier_shell is None:
        fourier_shell = FourierShell(L, 3, dtype, V.device)
    correlate_matrix, shell_size = covar_correlate(
        V, V, return_shell_size=True, fourier_shell=fourier_shell, chunked=chunked
    )
    return correlate_matrix.real * torch.outer(shell_size, shell_size)


def covar_fsc(eigenvecs1: torch.Tensor, eigenvecs2: torch.Tensor) -> torch.Tensor:
    """Compute Fourier shell correlation for covariance eigenvectors.

    Args:
        eigenvecs1: First set of eigenvectors
        eigenvecs2: Second set of eigenvectors

    Returns:
        Fourier shell correlation matrix
    """
    eigenvecs1_fft = centered_fft3(eigenvecs1)
    eigenvecs2_fft = centered_fft3(eigenvecs2)

    correlation = covar_correlate(eigenvecs1_fft, eigenvecs2_fft).real
    rpsd1 = covar_correlate(eigenvecs1_fft, eigenvecs1_fft).real
    rpsd2 = covar_correlate(eigenvecs2_fft, eigenvecs2_fft).real

    # Float32 precision might not have the dynamic range to compute the product (since they can be very large)
    # So we normalize the rpsd values by the zero frequency component and correct the normalization at the end
    bottom = (
        torch.sqrt((rpsd1 / rpsd1[0, 0]) * (rpsd2 / rpsd2[0, 0])) * torch.sqrt(rpsd1[0, 0]) * torch.sqrt(rpsd2[0, 0])
    )
    fsc = correlation / bottom

    return fsc
