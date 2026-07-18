from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from stratum.optimizer.ir._aggregation_ops import AggregateOp, GroupedDataframeOp
from stratum.optimizer.ir._base import IRNode
from stratum.optimizer.ir._dataframe_ops import (
    ApplyUDFOp,
    AssignMapOp,
    AssignOp,
    ColumnSelectorOp,
    ConcatOp,
    DatetimeConversionOp,
    DropOp,
    GetAttrProjectionOp,
    MapOp,
    MetadataOp,
    ProjectionOp,
    SelectionOp,
    SplitOp,
    SplitOutput,
    StringMethodOp,
)
from stratum.optimizer.ir._join_ops import JoinOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import (
    BaseEstimatorOp,
    BinOp,
    CallOp,
    ChoiceOp,
    GetAttrOp,
    GetItemOp,
    ImplOp,
    MethodCallOp,
    Op,
    SearchEvalOp,
    ValueOp,
    VariableOp,
    EstimatorOp,
    TransformerOp,
)
BackendName = str


"""Descriptor for one physical implementation of a plannable operator.

``op_type`` is the type the implementation is registered under: an *abstract
physical* op type for families already migrated to the physical layer (e.g.
``ReadCSV``), or a *logical* op type for families that still pass through
lowering unchanged (e.g. ``TransformerOp``). ``supports``/``cost``/``exec_mem``
form the fixed selector-facing API; ``impl_class`` names the concrete
``PhysicalOp`` the op is swapped to when this impl is chosen (identity preserved),
after which its ``on_impl_selected`` folds in any plan-time state.

``releases_gil`` / ``data_parallel`` are backend-specific capability hints,
stamped by the per-backend registration decorator. No selector consumes them
yet (the cost model is pending), but they expose to the parallelization planner
which impls can run concurrently (GIL released) and which already parallelize
internally (so it can avoid oversubscription)."""
@dataclass(frozen=True, slots=True)
class PhysicalImpl:
    op_type: type[IRNode]
    backend_name: BackendName
    input_format: str
    output_format: str
    supports: Callable[[IRNode], bool]
    cost: Callable[[IRNode, Any], float]
    exec_mem: Callable[[IRNode, Any], int]
    execute: Callable[[IRNode, str, list[Any]], Any]
    # Concrete PhysicalOp class the op is swapped to when this impl is chosen.
    impl_class: type | None = None
    # Backend capability hints (see the class docstring).
    releases_gil: bool = False
    data_parallel: bool = False


"""Operator family used to keep the registry extensible."""
@dataclass(frozen=True, slots=True)
class OperatorFamily:
    name: str
    op_types: tuple[type[IRNode], ...]
    default_backends: tuple[BackendName, ...] = ()
    notes: str = ""


"""A physical execution backend understood by the registry."""
@dataclass(frozen=True, slots=True)
class BackendSpec:
    name: str
    notes: str = ""


# FIXME: Only list the stratum's logical operators after compilation from skrub IR
# these will be replaced by the general physical operators (once they are here), and
# will be lowered to specific physical op implementations. Types leave this list as
# their family migrates to the physical layer (sources already have: DataSourceOp is
# lowered away and its physical types live in the "sources" family instead).
CURRENT_LOGICAL_OPERATOR_TYPES: tuple[type[Op], ...] = (
    AggregateOp,
    ApplyUDFOp,
    AssignMapOp,
    AssignOp,
    BaseEstimatorOp,
    BinOp,
    CallOp,
    ChoiceOp,
    ColumnSelectorOp,
    ConcatOp,
    DatetimeConversionOp,
    DropOp,
    EstimatorOp,
    GetAttrOp,
    GetAttrProjectionOp,
    GetItemOp,
    GroupedDataframeOp,
    ImplOp,
    JoinOp,
    MapOp,
    MetadataOp,
    MethodCallOp,
    NumericOp,
    ProjectionOp,
    SearchEvalOp,
    SelectionOp,
    SplitOp,
    SplitOutput,
    StringMethodOp,
    TransformerOp,
    ValueOp,
    VariableOp,
)


CURRENT_BACKENDS: tuple[BackendSpec, ...] = (
    BackendSpec("pandas", "Pandas dataframe implementation."),
    BackendSpec("polars", "Polars dataframe implementation."),
    BackendSpec("numpy", "NumPy array implementation."),
    BackendSpec("sklearn-skrub", "Existing sklearn/skrub implementation."),
    BackendSpec("rust", "Native Rust implementation selected like any other backend."),
)


CURRENT_OPERATOR_FAMILIES: tuple[OperatorFamily, ...] = (
    OperatorFamily(
        name="logical",
        op_types=CURRENT_LOGICAL_OPERATOR_TYPES,
        default_backends=tuple(backend.name for backend in CURRENT_BACKENDS),
        notes="Current logical IR surface; backends are attached later by the planner.",
    ),
)


def _unsupported_supports(op: Op) -> bool:
    return False


def _unsupported_cost(op: Op, stats: Any) -> float:
    raise NotImplementedError("No physical cost model has been registered for this operator yet.")


def _unsupported_exec_mem(op: Op, stats: Any) -> int:
    raise NotImplementedError("No execution-memory model has been registered for this operator yet.")


def _unsupported_execute(op: Op, mode: str, inputs: list[Any]) -> Any:
    raise NotImplementedError("No physical implementation has been registered for this operator yet.")


def _current_process_execute(op: Op, mode: str, inputs: list[Any]) -> Any:
    return op.process(mode, inputs)


def _placeholder_cost(op: Op, stats: Any) -> float:
    return 1.0


def _placeholder_exec_mem(op: Op, stats: Any) -> int:
    return 0


"""Container for physical implementations and their operator families."""
class PhysicalRegistry:
    def __init__(
        self,
        families: Iterable[OperatorFamily] = (),
        implementations: Iterable[PhysicalImpl] = (),
    ) -> None:
        self._families: list[OperatorFamily] = list(families)
        self._implementations: dict[type[IRNode], list[PhysicalImpl]] = {}
        self._implementations_by_backend: dict[BackendName, list[PhysicalImpl]] = {}
        for impl in implementations:
            self.register(impl)

    def register_family(self, family: OperatorFamily) -> None:
        self._families.append(family)

    def register(self, impl: PhysicalImpl) -> PhysicalImpl:
        self._implementations.setdefault(impl.op_type, []).append(impl)
        self._implementations_by_backend.setdefault(impl.backend_name, []).append(impl)
        return impl

    def families(self) -> tuple[OperatorFamily, ...]:
        return tuple(self._families)

    def op_types(self) -> tuple[type[IRNode], ...]:
        types: list[type[IRNode]] = []
        seen: set[type[IRNode]] = set()
        for family in self._families:
            for op_type in family.op_types:
                if op_type not in seen:
                    seen.add(op_type)
                    types.append(op_type)
        for op_type in self._implementations:
            if op_type not in seen:
                seen.add(op_type)
                types.append(op_type)
        return tuple(types)

    def candidates_for(
        self,
        op: type[IRNode] | IRNode,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        op_type = op if isinstance(op, type) else type(op)
        candidates = self._implementations.get(op_type, ())
        if backend_name is not None:
            candidates = [impl for impl in candidates if impl.backend_name == backend_name]
        return tuple(candidates)

    """Return the physical implementations available for a given operator."""
    def candidates_for_op(
        self,
        op: IRNode,
        backend_name: BackendName | None = None,
    ) -> tuple[PhysicalImpl, ...]:
        return self.candidates_for(op, backend_name=backend_name)

    def backends_for(self, op: type[IRNode] | IRNode) -> tuple[BackendName, ...]:
        return tuple(impl.backend_name for impl in self.candidates_for(op))

    def has_candidates(self, op: type[IRNode] | IRNode) -> bool:
        return len(self.candidates_for(op)) > 0

    def candidates_by_backend(self, backend_name: BackendName) -> tuple[PhysicalImpl, ...]:
        return tuple(self._implementations_by_backend.get(backend_name, ()))

    def empty(self) -> bool:
        return not self._implementations


# PhysicalImpl descriptors collected by the @physical_impl class decorator, in
# declaration order. build_default_physical_registry imports the exec modules
# (triggering decoration) and registers everything gathered here.
_DECORATED_IMPLS: list[PhysicalImpl] = []


def physical_impl(of: type[IRNode], backend: BackendName,
                  input_format: str = "frame", output_format: str = "frame",
                  releases_gil: bool = False, data_parallel: bool = False):
    """Class decorator registering a concrete PhysicalOp as an implementation.

    ``of`` is the (abstract) op type the class implements, e.g.::

        @physical_impl(of=ReadCSV, backend="pandas", input_format="value")
        class PandasReadCSV(ReadCSV): ...

    The selector-facing hooks (``supports``/``cost``/``exec_mem``) are taken from
    the class (PhysicalOp provides placeholder defaults); ``execute`` delegates to
    the op's own ``process``, which is concrete once the selection pass has swapped
    the op to this class. ``releases_gil`` / ``data_parallel`` are backend
    capability hints (usually supplied by a per-backend decorator, not here).

    Prefer the per-backend decorators (``rust_impl``, ``pandas_impl``, ...) which
    fix ``backend`` and its capability defaults; this is the general primitive
    they build on.
    """
    def deco(cls):
        impl = PhysicalImpl(
            op_type=of,
            backend_name=backend,
            input_format=input_format,
            output_format=output_format,
            supports=cls.supports,
            cost=cls.cost,
            exec_mem=cls.exec_mem,
            execute=_current_process_execute,
            impl_class=cls,
            releases_gil=releases_gil,
            data_parallel=data_parallel,
        )
        _DECORATED_IMPLS.append(impl)
        return cls
    return deco


# Per-backend registration decorators. Each fixes ``backend`` and that backend's
# default capability hints, so a single unified registry carries backend-specific
# information the planner can act on. An individual impl may override a default
# (e.g. a Rust kernel that is not data-parallel) via keyword argument.
def _backend_impl(backend: BackendName, releases_gil: bool, data_parallel: bool):
    def decorator(of: type[IRNode], input_format: str = "frame",
                  output_format: str = "frame",
                  releases_gil: bool = releases_gil,
                  data_parallel: bool = data_parallel):
        return physical_impl(of=of, backend=backend,
                             input_format=input_format, output_format=output_format,
                             releases_gil=releases_gil, data_parallel=data_parallel)
    return decorator


#: Native Rust kernels: release the GIL and parallelize internally (Rayon).
rust_impl = _backend_impl("rust", releases_gil=True, data_parallel=True)
#: Polars: releases the GIL and parallelizes internally.
polars_impl = _backend_impl("polars", releases_gil=True, data_parallel=True)
#: Pandas: single-threaded Python, holds the GIL.
pandas_impl = _backend_impl("pandas", releases_gil=False, data_parallel=False)
#: NumPy: C loops release the GIL but do not parallelize on their own.
numpy_impl = _backend_impl("numpy", releases_gil=True, data_parallel=False)
#: sklearn / skrub estimators: hold the GIL but can parallelize via n_jobs.
sklearn_skrub_impl = _backend_impl("sklearn-skrub", releases_gil=False, data_parallel=True)


def _register_current_estimator_impls(registry: PhysicalRegistry) -> None:
    for op_type in (TransformerOp, EstimatorOp):
        registry.register(
            PhysicalImpl(
                op_type=op_type,
                backend_name="sklearn-skrub",
                input_format="frame",
                output_format="frame",
                supports=lambda op: True,
                cost=_placeholder_cost,
                exec_mem=_placeholder_exec_mem,
                execute=_current_process_execute,
                data_parallel=True,
            )
        )


"""Create the default registry with every known implementation registered."""
def build_default_physical_registry() -> PhysicalRegistry:
    registry = PhysicalRegistry(families=CURRENT_OPERATOR_FAMILIES)

    # Imported lazily: the exec modules import back into this module (the
    # decorator / PhysicalImpl), so pulling them in at module level would cycle.
    # Importing each exec module triggers its @physical_impl registrations
    # (including the Rust kernels, now class-based @rust_impl impls).
    from stratum.optimizer.physical._source_execs import SOURCES_FAMILY  # noqa: F401
    from stratum.optimizer.physical import _transform_execs  # noqa: F401

    registry.register_family(SOURCES_FAMILY)
    for impl in _DECORATED_IMPLS:
        registry.register(impl)
    _register_current_estimator_impls(registry)
    return registry


_default_registry: PhysicalRegistry | None = None


def get_default_physical_registry() -> PhysicalRegistry:
    """Shared default registry, built once on first use.

    The implementation-selection pass consults this unless a registry is
    injected explicitly (tests, custom planners)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = build_default_physical_registry()
    return _default_registry
