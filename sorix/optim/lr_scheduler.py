"""
Learning rate schedulers for sorix optimizers.

Schedulers adjust the learning rate of each parameter group in an optimizer
following a policy. They do not touch gradients or parameters — they only
rewrite ``optimizer.param_groups[*]['lr']``.

A scheduler is advanced with ``scheduler.step()`` **after** ``optimizer.step()``,
once per epoch (not once per mini-batch)::

    optimizer = sorix.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = sorix.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    for epoch in range(100):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()          # computes gradients
            optimizer.step()         # applies them using the current lr
        scheduler.step()             # picks the lr for the next epoch

``ReduceLROnPlateau`` is the exception: it is metric-driven, so it is stepped
with the monitored value, ``scheduler.step(val_loss)``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sorix.optim.optim import Optimizer


class _LRScheduler:
    """
    Base class for all epoch-driven learning rate schedulers.

    Subclasses implement :meth:`get_lr`, which returns the learning rate for the
    current ``last_epoch`` as a **closed-form function of** ``base_lrs``. Being
    closed-form (rather than multiplying the optimizer's current lr) is what
    makes :meth:`load_state_dict` able to restore the learning rate exactly.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        last_epoch (int): Index of the last completed epoch. The scheduler
            resumes at ``last_epoch + 1``. Default: ``-1`` (fresh start, so the
            first epoch is 0).
    """

    def __init__(self, optimizer: "Optimizer", last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        # Store initial lrs from optimizer param_groups
        self.base_lrs: List[float] = [g["lr"] for g in optimizer.param_groups]
        self.step()

    def get_lr(self) -> List[float]:
        """Compute learning rates for the current epoch. Override in subclasses."""
        raise NotImplementedError

    def step(self) -> None:
        """Advance the scheduler by one epoch and update optimizer learning rates."""
        self.last_epoch += 1
        self._apply_lr()

    def _apply_lr(self) -> None:
        """Write the learning rates for the current epoch into the optimizer."""
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr

    def get_last_lr(self) -> List[float]:
        """Returns the last computed learning rate for each parameter group."""
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dict (excluding the optimizer)."""
        state = {k: v for k, v in self.__dict__.items() if k != "optimizer"}
        state["base_lrs"] = list(self.base_lrs)
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the scheduler state and re-applies the corresponding learning rate.

        Because :meth:`get_lr` is closed-form, restoring ``last_epoch`` and
        ``base_lrs`` is enough to recover the exact learning rate of the
        checkpointed epoch.
        """
        state_dict = dict(state_dict)
        state_dict.pop("optimizer", None)
        self.__dict__.update(state_dict)
        self.base_lrs = list(self.base_lrs)
        self._apply_lr()


class StepLR(_LRScheduler):
    """
    Decays the learning rate of each parameter group by ``gamma`` every
    ``step_size`` epochs, following the staircase schedule

    .. math:: \\eta_t = \\eta_0 \\cdot \\gamma^{\\lfloor t / s \\rfloor}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay. Must be >= 1.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch (int): Index of the last completed epoch. Default: -1.

    Example::

        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        # lr decays by 0.1× every 30 epochs
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if step_size < 1:
            raise ValueError(f"step_size must be >= 1, got {step_size}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        factor = self.gamma ** (self.last_epoch // self.step_size)
        return [base_lr * factor for base_lr in self.base_lrs]


class ExponentialLR(_LRScheduler):
    """
    Decays the learning rate of each parameter group by ``gamma`` every epoch,
    following

    .. math:: \\eta_t = \\eta_0 \\cdot \\gamma^{t}

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): Index of the last completed epoch. Default: -1.

    Example::

        scheduler = ExponentialLR(optimizer, gamma=0.95)
        # lr is multiplied by 0.95 each epoch
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        gamma: float,
        last_epoch: int = -1,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        factor = self.gamma ** self.last_epoch
        return [base_lr * factor for base_lr in self.base_lrs]


class CosineAnnealingLR(_LRScheduler):
    """
    Anneals the learning rate along a half cosine over ``T_max`` epochs:

    .. math::

        \\eta_t = \\eta_{\\min}
                + \\tfrac{1}{2}(\\eta_0 - \\eta_{\\min})
                  \\left(1 + \\cos\\left(\\frac{\\pi t}{T_{\\max}}\\right)\\right)

    So ``lr`` goes from ``base_lr`` at ``t = 0`` down to ``eta_min`` at
    ``t = T_max``.

    Note:
        The formula is periodic with period ``2 * T_max``. Stepping past
        ``T_max`` makes the learning rate **rise back** towards ``base_lr``
        (a "warm restart"). If you train for more than ``T_max`` epochs and do
        not want that, stop stepping the scheduler at ``T_max``.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations (half-period of the cosine).
            Must be >= 1.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): Index of the last completed epoch. Default: -1.

    Example::

        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if T_max < 1:
            raise ValueError(f"T_max must be >= 1, got {T_max}")
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        t = self.last_epoch
        T = self.T_max
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T)) / 2
            for base_lr in self.base_lrs
        ]


class ReduceLROnPlateau:
    """
    Reduces learning rate when a metric has stopped improving. Models often
    benefit from reducing the learning rate by a factor once learning stagnates.

    Unlike the epoch-driven schedulers, this one is metric-driven: call
    ``step(metric)`` with the monitored value after each validation pass.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): ``'min'`` or ``'max'``. In ``'min'`` mode, lr will be
            reduced when the quantity monitored has stopped decreasing;
            in ``'max'`` mode it will be reduced when the quantity has
            stopped increasing. Default: ``'min'``.
        factor (float): Factor by which the learning rate will be reduced.
            Default: 0.1.
        patience (int): Number of epochs with no improvement after which
            learning rate will be reduced. Default: 10.
        min_lr (float): A lower bound on the learning rate. Default: 0.
        threshold (float): Absolute improvement required to reset the patience
            counter. Default: 1e-4.

    Example::

        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        for epoch in range(epochs):
            train(...)
            val_loss = validate(...)
            scheduler.step(val_loss)
    """

    def __init__(
        self,
        optimizer: "Optimizer",
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        min_lr: float = 0.0,
        threshold: float = 1e-4,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        if factor >= 1.0:
            raise ValueError("factor must be < 1.0")

        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold

        self._best: Optional[float] = None
        self._num_bad_epochs: int = 0

    def _is_better(self, current: float) -> bool:
        if self._best is None:
            return True
        if self.mode == "min":
            return current < self._best - self.threshold
        return current > self._best + self.threshold

    def step(self, metrics: float) -> None:
        """Call after validation with the monitored metric value."""
        if self._is_better(metrics):
            self._best = metrics
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

        if self._num_bad_epochs >= self.patience:
            for group in self.optimizer.param_groups:
                new_lr = max(group["lr"] * self.factor, self.min_lr)
                group["lr"] = new_lr
            self._num_bad_epochs = 0

    def get_last_lr(self) -> List[float]:
        """Returns the current learning rate for each parameter group."""
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dict (excluding the optimizer)."""
        return {k: v for k, v in self.__dict__.items() if k != "optimizer"}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the scheduler state.

        Note:
            This scheduler mutates the optimizer's learning rate incrementally,
            so the restored learning rate is whatever the optimizer currently
            holds — load the optimizer's own ``state_dict`` alongside this one.
        """
        state_dict = dict(state_dict)
        state_dict.pop("optimizer", None)
        self.__dict__.update(state_dict)
