from stratum.optimizer.ir._dataframe_ops import ConcatOp
from stratum.optimizer.ir._numeric_ops import NumericOp
from stratum.optimizer.ir._ops import EstimatorOp, Op, TransformerOp
from stratum.optimizer.physical import (
    CURRENT_BACKENDS,
    CURRENT_LOGICAL_OPERATOR_TYPES,
    OperatorFamily,
    PhysicalImpl,
    PhysicalRegistry,
    build_default_physical_registry,
)
from stratum.optimizer.physical._transform_execs import (RustOneHotEncoder,
                                                        RustStringEncoder)


def test_default_registry_has_logical_surface_and_adapter_candidates():
    registry = build_default_physical_registry()

    assert not registry.empty()
    assert registry.op_types()
    assert ConcatOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert NumericOp in CURRENT_LOGICAL_OPERATOR_TYPES
    assert "rust" in {backend.name for backend in CURRENT_BACKENDS}
    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")
    sklearn_candidates = registry.candidates_for(TransformerOp, backend_name="sklearn-skrub")
    assert len(rust_candidates) == 2
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert len(sklearn_candidates) == 1
    assert len(registry.candidates_for(EstimatorOp, backend_name="sklearn-skrub")) == 1


def test_rust_kernels_are_class_based_impls():
    # Every Rust kernel is a class-based @rust_impl keyed on the logical
    # TransformerOp; there is no separate Rust registration list.
    registry = build_default_physical_registry()

    rust_candidates = registry.candidates_for(TransformerOp, backend_name="rust")

    assert len(rust_candidates) == 2
    assert all(candidate.backend_name == "rust" for candidate in rust_candidates)
    assert {c.impl_class for c in rust_candidates} == {RustStringEncoder, RustOneHotEncoder}


def test_registry_registers_and_queries_impls_by_logical_type():
    registry = PhysicalRegistry()

    class DummyOp(Op):
        pass

    pandas_impl = PhysicalImpl(
        op_type=DummyOp,
        backend_name="pandas",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 1.0,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("concat", mode, len(inputs)),
    )
    rust_impl = PhysicalImpl(
        op_type=DummyOp,
        backend_name="rust",
        input_format="frame",
        output_format="frame",
        supports=lambda op: isinstance(op, DummyOp),
        cost=lambda op, stats: 0.5,
        exec_mem=lambda op, stats: 1,
        execute=lambda op, mode, inputs: ("rust-concat", mode, len(inputs)),
    )

    registry.register(pandas_impl)
    registry.register(rust_impl)

    assert registry.candidates_for(DummyOp) == (pandas_impl, rust_impl)
    assert registry.candidates_for_op(DummyOp()) == (pandas_impl, rust_impl)
    assert registry.candidates_for(DummyOp, backend_name="rust") == (rust_impl,)
    assert registry.backends_for(DummyOp) == ("pandas", "rust")
    assert registry.candidates_by_backend("pandas") == (pandas_impl,)
    assert registry.candidates_by_backend("rust") == (rust_impl,)


def test_register_family_tracks_known_logical_types():
    registry = PhysicalRegistry()

    family = OperatorFamily(
        name="custom",
        op_types=(ConcatOp,),
        default_backends=("pandas",),
    )
    registry.register_family(family)

    assert registry.families() == (family,)
    assert ConcatOp in registry.op_types()
