from collections import deque
from typing import Any, Callable, Dict, Optional

import torch
from torch.optim.optimizer import Optimizer


class BlockNewtonOptimizer(torch.optim.Optimizer):
    """Implements a Newton optimizer with backtracking line search.

    Assumes hessian is block diagonal with respect to the parameters and batch (first dim) of
    tensor.

    Attributes:
        param_groups: List of parameter groups
        defaults: Default parameter values
    """

    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        max_ls_steps: int = 10,
        c: float = 1e-4,
        beta: float = 0.1,
        damping: float = 1e-6,
        line_search: bool = True,
        step_size_limit: Optional[float] = None,
    ) -> None:
        """Initialize the Block Newton optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1.0)
            max_ls_steps: Maximum number of line search steps (default: 10)
            c: Armijo condition constant (default: 1e-4)
            beta: Line search step size reduction factor (default: 0.1)
            damping: Hessian damping factor for numerical stability (default: 1e-6)
            line_search: Whether to use backtracking line search (default: True)
            step_size_limit: Maximum step size limit (optional)
        """
        defaults = dict(
            lr=lr,
            max_ls_steps=max_ls_steps,
            c=c,
            beta=beta,
            damping=damping,
            line_search=line_search,
            step_size_limit=step_size_limit,
        )
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value

        Raises:
            ValueError: If closure is None
        """
        loss = None
        if closure is None:
            raise ValueError("Newton optimizer requires a closure to reevaluate the model.")

        with torch.enable_grad():
            loss = closure()
            loss.sum().backward(create_graph=True)

        for group in self.param_groups:
            lr = group["lr"]
            c = group["c"]
            beta = group["beta"]
            max_ls_steps = group["max_ls_steps"]
            damping = group["damping"]
            line_search = group["line_search"]
            step_size_limit = group["step_size_limit"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                orig_param = param.data.clone()
                flat_aggregated_grad = param.grad.sum(dim=0).view(-1, 1)
                n = flat_aggregated_grad.shape[0]
                batch_size = param.shape[0]

                # Compute Hessian
                hessian = torch.zeros(
                    (n,) + param.shape, dtype=flat_aggregated_grad.dtype, device=flat_aggregated_grad.device
                )
                for i in range(n):
                    hessian[i] = torch.autograd.grad(flat_aggregated_grad[i], param, retain_graph=True)[0]

                hessian = hessian.reshape(n, batch_size, n).transpose(0, 1)
                # Damping for stability
                hessian = hessian + damping * torch.eye(n, device=param.device).unsqueeze(0)

                # Compute Newton step
                step_dir = torch.linalg.solve(hessian, param.grad.view(batch_size, -1))
                if step_size_limit is not None:
                    step_dir = torch.clamp(step_dir, -step_size_limit, step_size_limit)

                # Backtracking line search
                if line_search:
                    alpha = lr
                    alpha_step_taken = torch.zeros(batch_size, device=param.device)
                    for _ in range(max_ls_steps):
                        param.data = orig_param - alpha * step_dir.view_as(param.data)

                        with torch.enable_grad():
                            self.zero_grad()
                            trial_loss = closure()
                            trial_loss.sum().backward(create_graph=True)

                        alpha_step_taken[
                            (
                                (
                                    trial_loss
                                    <= loss - c * alpha * torch.norm(param.grad.view(batch_size, -1), dim=1) ** 2
                                ).to(torch.int)
                                + (alpha_step_taken == 0)
                            )
                            == 2
                        ] = alpha

                        alpha *= beta

                    param.data = orig_param - step_dir.view_as(param.data) * alpha_step_taken.reshape(
                        (-1,) + (param.data.ndim - 1) * (1,)
                    )
                else:
                    # No line search, take the full step
                    param.data = orig_param - lr * step_dir.view_as(param.data)

        return loss


class BlockwiseLBFGS(Optimizer):
    """Blockwise Limited-memory BFGS optimizer.

    Implements L-BFGS optimization for block-diagonal parameters, where each block
    is optimized independently using its own history of gradients and parameter updates.

    Attributes:
        param_groups: List of parameter groups
        defaults: Default parameter values
        history_size: Maximum number of history vectors to store
        state: Internal state for tracking optimization history
    """

    def __init__(
        self, params: Any, lr: float = 1.0, history_size: int = 10, step_size_limit: Optional[float] = None
    ) -> None:
        """Initialize the Blockwise L-BFGS optimizer.

        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1.0)
            history_size: Maximum number of history vectors to store (default: 10)
            step_size_limit: Maximum step size limit (optional)

        Raises:
            ValueError: If learning rate is not positive
        """
        if lr <= 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, step_size_limit=step_size_limit)
        super(BlockwiseLBFGS, self).__init__(params, defaults)

        self.history_size = history_size

        # Each param is assumed to be a single nn.Embedding
        self.state = self._init_state()

    def _init_state(self) -> Dict[str, Dict[int, Any]]:
        """Initialize the optimizer state.

        Returns:
            Dictionary containing state for tracking optimization history
        """
        return {
            "s_history": {},  # row_id -> deque of s vectors (delta_x)
            "y_history": {},  # row_id -> deque of y vectors (delta_grad)
            "prev_x": {},  # row_id -> previous x vector
            "prev_grad": {},  # row_id -> previous grad vector
        }

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss

        Returns:
            The loss value
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        param = group["params"][0]
        lr = group["lr"]
        step_size_limit = group["step_size_limit"]
        weight = param  # n × d
        grad = param.grad  # same shape

        if grad is None:
            return loss

        # Get indices of rows that were used (i.e., got non-zero grad)
        if hasattr(param, "grad_indices"):  # optional if you're tracking which indices were used
            indices = param.grad_indices
        else:
            indices = grad._indices()[0] if grad.is_sparse else torch.nonzero(grad.norm(dim=1), as_tuple=True)[0]

        for i in indices:
            i = i.item()
            x_i = weight[i]
            g_i = grad[i]

            # History storage
            s_hist = self.state["s_history"].setdefault(i, deque(maxlen=self.history_size))
            y_hist = self.state["y_history"].setdefault(i, deque(maxlen=self.history_size))
            x_prev = self.state["prev_x"].get(i, None)
            g_prev = self.state["prev_grad"].get(i, None)

            # Save current for next time
            self.state["prev_x"][i] = x_i.detach().clone()
            self.state["prev_grad"][i] = g_i.detach().clone()

            if x_prev is not None and g_prev is not None:
                s = x_i.detach() - x_prev
                y = g_i.detach() - g_prev
                if torch.dot(s, y) > 1e-10:  # ensure curvature condition
                    s_hist.append(s)
                    y_hist.append(y)

            # L-BFGS two-loop recursion
            q = g_i.detach()
            alphas = []
            rho_list = []

            for s, y in reversed(list(zip(s_hist, y_hist))):
                rho = 1.0 / torch.dot(y, s)
                alpha = rho * torch.dot(s, q)
                q = q - alpha * y
                alphas.append(alpha)
                rho_list.append(rho)

            # Initial Hessian approximation: scalar times identity
            if s_hist:
                y_last = y_hist[-1]
                s_last = s_hist[-1]
                H0 = torch.dot(s_last, y_last) / torch.dot(y_last, y_last)
            else:
                H0 = 1.0

            r = H0 * q

            for s, y, alpha, rho in zip(s_hist, y_hist, reversed(alphas), reversed(rho_list)):
                beta = rho * torch.dot(y, r)
                r = r + s * (alpha - beta)

            step = lr * r
            if step_size_limit is not None:
                step = torch.clamp(step, -step_size_limit, step_size_limit)
            # Apply update
            weight[i].data -= step
        return loss
