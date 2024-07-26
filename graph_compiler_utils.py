from functools import partial
from typing import Any

import torch
from torch._decomp.decompositions import native_layer_norm_backward

# from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
# from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec

aten = torch.ops.aten  # pyre-ignore


def _foreach_add_decomp(self, other, alpha=1):
    self_updated = aten._foreach_add.List(self, other, alpha=alpha)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_unaop_decomp(op, self):
    self_updated = op(self)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_list_decomp(op, self, other):
    self_updated = op(self, other)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_binop_scalar_decomp(op, self, scalar=1):
    self_updated = op(self, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _foreach_addcop_scalar_decomp(op, self, tensor1, tensor2, scalar=1):
    self_updated = op(self, tensor1, tensor2, scalar)
    for s, s_u in zip(self, self_updated):
        s.copy_(s_u)


def _fused_adam_decomp(
    self,
    grads,
    exp_avgs,
    exp_avg_sqs,
    max_exp_avg_sqs,
    state_steps,
    *,
    lr=1,
    beta1=1,
    beta2=1,
    weight_decay=1,
    eps=1,
    amsgrad=True,
    maximize=True,
    grad_scale=None,
    found_inf=None,
):
    orig_tuple = (self, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs)
    updated_tuple = aten._fused_adam.default(
        self,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
        eps=eps,
        amsgrad=amsgrad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

    for idx, (orig, updated) in enumerate(zip(orig_tuple, updated_tuple)):
        if idx == 1:
            # skip gradient copying as we don't need to copy gradients back
            continue
        for o, u in zip(orig, updated):
            o.copy_(u)


SPMD_DECOMP_TABLE = {
    aten._foreach_add_.List: _foreach_add_decomp,
    aten._foreach_add_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_add.Scalar
    ),
    aten._foreach_addcdiv_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcdiv.Scalar
    ),
    aten._foreach_addcmul_.Scalar: partial(
        _foreach_addcop_scalar_decomp, aten._foreach_addcmul.Scalar
    ),
    aten._foreach_div_.List: partial(
        _foreach_binop_list_decomp, aten._foreach_div.List
    ),
    aten._foreach_mul_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_mul.Scalar
    ),
    aten._foreach_div_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_div.Scalar
    ),
    aten._foreach_neg_.default: partial(
        _foreach_unaop_decomp, aten._foreach_neg.default
    ),
    aten._foreach_reciprocal_.default: partial(
        _foreach_unaop_decomp, aten._foreach_reciprocal.default
    ),
    aten._foreach_sqrt_.default: partial(
        _foreach_unaop_decomp, aten._foreach_sqrt.default
    ),
    aten._foreach_sub_.Scalar: partial(
        _foreach_binop_scalar_decomp, aten._foreach_sub.Scalar
    ),
    aten._fused_adam_.default: _fused_adam_decomp,
    aten.native_layer_norm_backward.default: native_layer_norm_backward,
}


def sep(x: torch.Tensor) -> torch.Tensor:
    return x


def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad


separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


# def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
#     (x,) = op_schema.args_schema
#     assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

#     return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))


# @register_prop_rule(torch.ops.separator.sep.default)
# def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
#     return _identity_prop_rule(op_schema)


# @register_prop_rule(torch.ops.separator.sep_backward.default)
# def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
#     return _identity_prop_rule(op_schema)


class SEPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep_backward(grad_x)


# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")
