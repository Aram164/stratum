"""Physical implementations of ``SelectionOp`` (relational row selection).

Same-shape backend-variant family. The MASK kind evaluates a backend-agnostic
``ColumnExpr`` predicate; method-based kinds (dropna, head, ...) map to a
per-backend method name. The pandas ``query()`` fast path is a plan-time
decision (``ctx.pandas_query``) folded into instance state, so ``process`` reads
data, never flags.
"""
from __future__ import annotations

from stratum.optimizer.ir._base import _resolve_args, _resolve_kwargs
from stratum.optimizer.ir._column_expr import EvalContext
from stratum.optimizer.ir._selection_ops import (
    SelectionKind, SelectionOp, _SELECTION_PANDAS_METHOD, _SELECTION_POLARS_METHOD)
from stratum.optimizer.physical._physical_ops import PhysicalOp
from stratum.optimizer.physical._registry import physical_impl


@physical_impl(of=SelectionOp, backend="pandas")
class PandasSelectionOp(SelectionOp, PhysicalOp):
    def on_impl_selected(self, ctx) -> None:
        # Whether to route MASK selections through DataFrame.query() is a plan-time
        # decision; fold it into instance state.
        self.use_query = ctx.pandas_query

    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        if self.kind is SelectionKind.MASK:
            ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
            if getattr(self, "use_query", False):
                params = {}
                query = self.predicate.to_pandas_query(params)
                # None when the predicate isn't query-expressible (an OperandLeaf
                # or str accessor); fall through to boolean masking in that case.
                if query is not None:
                    return _obj.query(query, local_dict=params)
            return _obj[self.predicate.to_pandas(ctx)]
        _args = _resolve_args(self.args, inputs) if self.args else []
        _kwargs = _resolve_kwargs(self.kwargs, inputs) if self.kwargs else {}
        method = _SELECTION_PANDAS_METHOD.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"SelectionOp.process is not implemented for kind {self.kind.name}.")
        return getattr(_obj, method)(*_args, **_kwargs)


@physical_impl(of=SelectionOp, backend="polars")
class PolarsSelectionOp(SelectionOp, PhysicalOp):
    def process(self, mode: str, inputs: list):
        _obj = inputs[0]
        if self.kind is SelectionKind.MASK:
            ctx = EvalContext(frame=_obj, inputs=inputs, mode=mode)
            return _obj.filter(self.predicate.to_polars(ctx))
        _args = _resolve_args(self.args, inputs) if self.args else []
        _kwargs = _resolve_kwargs(self.kwargs, inputs) if self.kwargs else {}
        method = _SELECTION_POLARS_METHOD.get(self.kind)
        if method is None:
            raise NotImplementedError(
                f"SelectionOp.process is not implemented for kind {self.kind.name} "
                f"on the Polars backend.")
        return getattr(_obj, method)(*_args, **_kwargs)
