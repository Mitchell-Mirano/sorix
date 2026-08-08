"""Tests for sorix.optim.lr_scheduler."""
import math
import numpy as np
import pytest
import sorix
from sorix.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
)


def _make_optimizer(lr=1e-3):
    """Helper: single Linear layer + Adam optimizer."""
    layer = sorix.nn.Linear(4, 4)
    return sorix.optim.Adam(layer.parameters(), lr=lr)


# ── StepLR ────────────────────────────────────────────────────────────────────

def test_steplr_no_decay_before_step_size():
    opt = _make_optimizer(lr=1.0)
    sched = StepLR(opt, step_size=3, gamma=0.1)
    for _ in range(2):   # epochs 1 and 2 → no decay yet
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0)


def test_steplr_decay_at_step_size():
    opt = _make_optimizer(lr=1.0)
    sched = StepLR(opt, step_size=3, gamma=0.1)
    for _ in range(3):   # epoch 3 → first decay
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_steplr_multiple_decays():
    opt = _make_optimizer(lr=1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    for _ in range(4):   # 2 decays → 1.0 * 0.5 * 0.5 = 0.25
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.25)


# ── ExponentialLR ──────────────────────────────────────────────────────────────

def test_exponential_lr_decreases_each_epoch():
    opt = _make_optimizer(lr=1.0)
    gamma = 0.9
    sched = ExponentialLR(opt, gamma=gamma)
    prev_lr = 1.0
    for _ in range(5):
        sched.step()
        current_lr = opt.param_groups[0]["lr"]
        assert current_lr == pytest.approx(prev_lr * gamma)
        prev_lr = current_lr


def test_exponential_lr_formula():
    opt = _make_optimizer(lr=0.1)
    sched = ExponentialLR(opt, gamma=0.95)
    for k in range(1, 6):
        sched.step()
    expected = 0.1 * (0.95 ** 5)
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-5)


# ── CosineAnnealingLR ──────────────────────────────────────────────────────────

def test_cosine_annealing_starts_at_base_lr():
    opt = _make_optimizer(lr=0.1)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.0)
    # After construction (__init__ calls step() once → epoch 0)
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1, rel=1e-5)


def test_cosine_annealing_reaches_eta_min():
    opt = _make_optimizer(lr=0.1)
    sched = CosineAnnealingLR(opt, T_max=10, eta_min=0.001)
    for _ in range(10):   # epoch 10 → cos(π) = -1 → eta_min
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.001, rel=1e-4)


def test_cosine_annealing_formula():
    base_lr = 0.1
    eta_min = 0.0
    T_max = 50
    opt = _make_optimizer(lr=base_lr)
    sched = CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
    for _ in range(25):
        sched.step()
    expected = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * 25 / T_max)) / 2
    assert opt.param_groups[0]["lr"] == pytest.approx(expected, rel=1e-5)


# ── ReduceLROnPlateau ──────────────────────────────────────────────────────────

def test_reduce_on_plateau_reduces_after_patience():
    opt = _make_optimizer(lr=0.1)
    sched = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    sched.step(0.5)   # baseline — sets best
    for _ in range(3):
        sched.step(0.5)   # no improvement — 3 bad epochs → triggers reduction
    assert opt.param_groups[0]["lr"] == pytest.approx(0.05)



def test_reduce_on_plateau_improves_resets_counter():
    opt = _make_optimizer(lr=0.1)
    sched = ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    sched.step(1.0)   # baseline
    sched.step(0.5)   # improvement → reset
    sched.step(0.5)   # no improvement (1)
    sched.step(0.5)   # no improvement (2)
    # Only 2 bad epochs since reset → no decay yet
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)


def test_reduce_on_plateau_min_lr_floor():
    opt = _make_optimizer(lr=0.01)
    sched = ReduceLROnPlateau(opt, mode="min", patience=1, factor=0.1, min_lr=0.001)
    for _ in range(5):
        sched.step(1.0)   # never improves
    assert opt.param_groups[0]["lr"] >= 0.001


# ── State dict round-trip ──────────────────────────────────────────────────────

def test_steplr_state_dict_roundtrip():
    opt = _make_optimizer(lr=1.0)
    sched = StepLR(opt, step_size=3, gamma=0.5)
    for _ in range(3):
        sched.step()
    state = sched.state_dict()
    assert state["last_epoch"] == 3

    # Restore into a fresh scheduler
    opt2 = _make_optimizer(lr=1.0)
    sched2 = StepLR(opt2, step_size=3, gamma=0.5)
    sched2.load_state_dict(state)
    assert sched2.last_epoch == 3


@pytest.mark.parametrize("factory", [
    lambda opt: StepLR(opt, step_size=2, gamma=0.5),
    lambda opt: ExponentialLR(opt, gamma=0.9),
    lambda opt: CosineAnnealingLR(opt, T_max=20, eta_min=1e-4),
])
def test_state_dict_restores_learning_rate(factory):
    """Restoring a checkpoint must reproduce the lr, not just the epoch counter."""
    opt = _make_optimizer(lr=1.0)
    sched = factory(opt)
    for _ in range(5):
        sched.step()
    expected_lr = opt.param_groups[0]["lr"]
    state = sched.state_dict()

    opt2 = _make_optimizer(lr=1.0)
    sched2 = factory(opt2)
    sched2.load_state_dict(state)
    assert opt2.param_groups[0]["lr"] == pytest.approx(expected_lr)

    # ...and training must continue along the same schedule.
    sched.step()
    sched2.step()
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])


def test_state_dict_does_not_alias_base_lrs():
    opt = _make_optimizer(lr=1.0)
    sched = StepLR(opt, step_size=2, gamma=0.5)
    state = sched.state_dict()
    sched.base_lrs[0] = 99.0
    assert state["base_lrs"][0] == pytest.approx(1.0)


# ── Argument validation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("factory, match", [
    (lambda opt: StepLR(opt, step_size=0), "step_size must be >= 1"),
    (lambda opt: StepLR(opt, step_size=2, gamma=0.0), "gamma must be > 0"),
    (lambda opt: ExponentialLR(opt, gamma=-1.0), "gamma must be > 0"),
    (lambda opt: CosineAnnealingLR(opt, T_max=0), "T_max must be >= 1"),
    (lambda opt: ReduceLROnPlateau(opt, mode="lowest"), "mode must be"),
    (lambda opt: ReduceLROnPlateau(opt, factor=1.0), "factor must be < 1.0"),
])
def test_invalid_arguments_raise(factory, match):
    opt = _make_optimizer(lr=0.1)
    with pytest.raises(ValueError, match=match):
        factory(opt)


def test_get_last_lr():
    opt = _make_optimizer(lr=0.01)
    sched = StepLR(opt, step_size=5, gamma=0.1)
    lrs = sched.get_last_lr()
    assert len(lrs) == len(opt.param_groups)
    assert lrs[0] == pytest.approx(0.01)
