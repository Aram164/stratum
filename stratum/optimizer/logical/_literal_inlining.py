"""Inline scalar ``ValueOp``s into consumers that can hold the literal directly."""
from collections import defaultdict
from typing import Any

from stratum.optimizer._op_utils import topological_iterator
from stratum.optimizer.logical._base import IRNode, rebuild_dict
from stratum.optimizer.logical._dataframe_ops import ConcatOp
from stratum.optimizer.logical._ops import BinOp, CallOp, GetItemOp, MethodCallOp, Op, OperandRef, ValueOp, remap_operand_refs


def _is_literal(value: Any) -> bool:
    return type(value) in (type(None), bool, int, float, str, bytes)


def _inlinable(out: Op, pos: int) -> bool:
    if isinstance(out, ConcatOp):
        if isinstance(out.axis, OperandRef) and out.axis.k == pos:
            param_type = type(out.inputs[pos].value) 
            return param_type in (str, int)
    if not isinstance(out, (CallOp, MethodCallOp, GetItemOp, BinOp)):
        return False
    if isinstance(out, (MethodCallOp, GetItemOp)):
        return pos != 0
    if isinstance(out, CallOp) and getattr(out.func, "__module__", None) == "numpy":
        return pos != 0
    if isinstance(out, BinOp):
        # Keep a graph input for numeric extraction / algebraic rewrites: never
        # inline away every input, and never hide a shared producer's identity
        # (e.g. ``x * x``) behind a literal at one of its BinOp consumers.
        if len(out.inputs[pos].outputs) != 1:
            return False
        return not all(isinstance(i, ValueOp) and _is_literal(i.value) for i in out.inputs)
    return True


def literal_inlining(root: Op) -> Op:
    records: dict[Op, set[int]] = defaultdict(set)
    for op in list(topological_iterator(root)):
        if not isinstance(op, ValueOp) or not _is_literal(op.value) or op.is_X or op.is_y:
            continue
        for out in op.outputs:
            for pos, input_op in enumerate(out.inputs):
                if input_op is op and _inlinable(out, pos) and _check_for_constant(op, out, pos):
                    records[out].add(pos)
                    if isinstance(out, GetItemOp):
                        out.update_name()
    _rewire_edges(records)
    return root


def _rewire_edges(records: dict[Op, set[int]]) -> None:
    for op, removed in records.items():
        removed_set = set(removed)
        new_inputs = [input_op for i, input_op in enumerate(op.inputs) if i not in removed_set]
        mapping, offset = {}, 0
        for i, input_op in enumerate(op.inputs):
            if i in removed_set:
                if not any(inp is input_op for inp in new_inputs):
                    input_op.outputs = [out for out in input_op.outputs if out is not op]
                offset += 1
            else:
                mapping[i] = i - offset
        for field in op.fields:
            param = getattr(op, field)
            new_param = remap_operand_refs(param, mapping)
            if param is not new_param:
                setattr(op, field, new_param)
        op.inputs = new_inputs


def _check_for_constant(value_node: ValueOp, out: IRNode, pos: int) -> bool:
    changed_any = False
    for field in out.fields:
        param = getattr(out, field)
        changed, new_param = _inline_literal(param, pos, value_node.value)
        if changed:
            setattr(out, field, new_param)
            changed_any = True
    return changed_any


def _inline_literal(param: Any, pos: int, literal: Any) -> tuple[bool, Any]:
    if isinstance(param, OperandRef):
        return (True, literal) if param.k == pos else (False, param)
    if isinstance(param, tuple):
        results = [_inline_literal(v, pos, literal) for v in param]
        changed = any(flag for flag, _ in results)
        return changed, tuple(value for _, value in results) if changed else param
    if isinstance(param, list):
        results = [_inline_literal(v, pos, literal) for v in param]
        changed = any(flag for flag, _ in results)
        return changed, [value for _, value in results] if changed else param
    if isinstance(param, dict):
        results = {k: _inline_literal(v, pos, literal) for k, v in param.items()}
        changed = any(flag for flag, _ in results.values())
        return changed, rebuild_dict(param, ((k, result[1]) for k, result in results.items())) if changed else param
    if isinstance(param, slice):
        results = [_inline_literal(v, pos, literal) for v in (param.start, param.stop, param.step)]
        changed = any(flag for flag, _ in results)
        return changed, slice(*(value for _, value in results)) if changed else param
    return False, param
