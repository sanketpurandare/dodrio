import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from copy import copy
# We need to import _functional_collectives to trigger op registration
import torch.distributed._functional_collectives
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from typing import Any, Callable, List, Dict
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec

def sep(x: torch.Tensor) -> torch.Tensor:
    return x

def sep_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad


separator_lib = torch.library.Library("separator", "DEF")
separator_lib.define("sep(Tensor x) -> Tensor")
separator_lib.impl("sep", sep, "CompositeExplicitAutograd")
separator_lib.define("sep_backward(Tensor x) -> Tensor")
separator_lib.impl("sep_backward", sep_backward, "CompositeExplicitAutograd")


def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))


@register_prop_rule(torch.ops.separator.sep.default)
def _prop_sepm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


@register_prop_rule(torch.ops.separator.sep_backward.default)
def _prop_sepm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


class SEPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        return torch.ops.separator.sep_backward(grad_x)

# Dummy op used by data parallel to tag gradients.
_spmd_lib_def = torch.library.Library("_spmd", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("_spmd", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")

@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """
    This is a helper function that tags the gradient of the parameters
    with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """

    tagging_hooks:List[RemovableHandle] = []
    try:
        for p in params.values():
            h = p.register_hook(lambda grad: torch.ops._spmd.tag_grad(grad))
            tagging_hooks.append(h)
        yield
    finally:
        # remove those hooks after tracing
        for h in tagging_hooks:
            h.remove()

@contextmanager
def _rematerialize_optimizer(
    opt: torch.optim.Optimizer,
    named_states: Dict[str, Any],
    params: Dict[str, nn.Parameter],
):
    assert opt is not None

    # update opt.state with proxy tensors
    orig_states = copy(opt.state)
    for n in named_states:
        # opt.state's key type is string, but optimizer uses Parameter as keys
        opt.state[params[n]] = named_states[n]  # type: ignore[index]

    # FIXME: support multiple parameter groups
    param_group = opt.param_groups[0]
    orig_params = param_group["params"]
    param_group["params"] = params.values()

    try:
        yield
    finally:
        param_group["params"] = orig_params
        opt.state = orig_states

def compile(func: Callable, *args: Any, **kwargs: Any):
    # 1. Extract nn.Module and Optimizer from args and kwargs
    mod, opt = None, None
    for arg in pytree.tree_flatten(list(args) + list(kwargs.values()))[0]:
        if isinstance(arg, nn.Module):
            assert mod is None, "Only support single nn.Module for now"
            mod = arg
        if isinstance(arg, optim.Optimizer):
            assert opt is None, "Only support single Optimizer for now"
            opt = arg
    assert mod is not None, "Couldn't find nn.Module instances from the arguments."

    # 2. Trace the stateless version of the train_step
    params = dict(mod.named_parameters(remove_duplicate=False))
    buffers = dict(mod.named_buffers(remove_duplicate=False))

    named_states= {}
    # Pass named_states instead of opt.state to stateless_func, because
    # the later uses nn.Parameter as key. During tracing, we need to
    # make sure optimizers can find the states using proxy tensors.
    for n, p in params.items():
        if p in opt.state:
            # opt.state's key type is string, but optimizer uses 
            # Parameter as keys
            named_states[n] = opt.state[p]
    # Lift states and parameters as function arguments so that make_fx 
    # can trace operations applied to them
            
    def stateless_func(func, params, buffers, named_states, args, kwargs):
        with stateless._reparametrize_module(mod, {**params, **buffers}
        ), _rematerialize_optimizer(opt, named_states, params
        ) if opt else nullcontext():
            #Installing hooks onto gradients to identify the gradients.
            with gradients_tagging(params):
                ret = func(*args, **kwargs)

            # the return value of the function must be the original return value
            # updated paramaters and updated optimizer states
            return ret, list(mod.parameters()), list(named_states.values()) 
            
        






