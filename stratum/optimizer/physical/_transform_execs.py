"""Physical transformer operators (native Rust kernels).

The native Rust transformer kernels register as ordinary class-based physical
impls (``@rust_impl``), keyed on the logical ``TransformerOp`` they refine --
there is no separate Rust registry. Implementation selection swaps a supported
``TransformerOp`` to the Rust impl class in place, and its ``on_impl_selected``
swaps the op's estimators for the Rust adapter at plan time, so ``process`` runs
the Rust kernel with no run-time decision left.
"""
from __future__ import annotations

from typing import Any

from stratum.adapters.one_hot_encoder import (RustyOneHotEncoder,
                                             supports_rust_one_hot_encoder)
from stratum.adapters.string_encoder import (RustyStringEncoder,
                                             supports_rust_string_encoder)
from stratum.optimizer.ir._ops import TransformerOp
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import rust_impl


@rust_impl(of=TransformerOp)
class RustStringEncoder(TransformerOp, PhysicalOp):
    """Native Rust string encoder: swaps in the ``RustyStringEncoder`` adapter at
    plan time, so ``process`` runs the Rust kernel with no run-time decision left."""

    @classmethod
    def supports(cls, op: TransformerOp) -> bool:
        supported, _ = supports_rust_string_encoder(op.original_estimator)
        return supported

    def on_impl_selected(self, ctx: Any) -> None:
        self.original_estimator = _as_rusty_string_encoder(self.original_estimator)
        self.estimator = _as_rusty_string_encoder(self.estimator)


def _as_rusty_string_encoder(estimator) -> RustyStringEncoder:
    """Adapt a skrub ``StringEncoder`` to the Rust drop-in, forcing the Rust path."""
    if isinstance(estimator, RustyStringEncoder):
        rusty = estimator
    else:
        rusty = RustyStringEncoder(**estimator.get_params(deep=False))
    rusty._stratum_force_rust = True
    return rusty


@rust_impl(of=TransformerOp, output_format="matrix")
class RustOneHotEncoder(TransformerOp, PhysicalOp):
    """Native Rust one-hot encoder: swaps in ``RustyOneHotEncoder`` at plan time."""

    @classmethod
    def supports(cls, op: TransformerOp) -> bool:
        supported, _ = supports_rust_one_hot_encoder(op.original_estimator)
        return supported

    def on_impl_selected(self, ctx: Any) -> None:
        self.original_estimator = _as_rusty_one_hot_encoder(self.original_estimator)
        self.estimator = _as_rusty_one_hot_encoder(self.estimator)


def _as_rusty_one_hot_encoder(estimator) -> RustyOneHotEncoder:
    """Adapt a sklearn ``OneHotEncoder`` to the Rust drop-in, forcing the Rust path."""
    if isinstance(estimator, RustyOneHotEncoder):
        rusty = estimator
    else:
        rusty = RustyOneHotEncoder(**estimator.get_params(deep=False))
    rusty._stratum_force_rust = True
    return rusty
