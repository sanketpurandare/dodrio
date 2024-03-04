import torch
import torch.distributed as dist
from contextlib import contextmanager, nullcontext
from copy import copy
from functools import partial, wraps
from dataclasses import dataclass

# We need to import _functional_collectives to trigger op registration
import torch.distributed._functional_collectives
from torch.distributed._functional_collectives import all_reduce
import torch.nn as nn
import torch.optim as optim
import torch.utils._pytree as pytree
from torch import fx
from torch.distributed._spmd.api import SPMD_DECOMP_TABLE
from torch.fx.graph import _PyTreeCodeGen, _PyTreeInfo, CodeGen
from typing import Any, Callable, List, Dict, Union, Optional
from torch.nn.utils import stateless
from torch.utils.hooks import RemovableHandle
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from graph_profiler import GraphProfiler


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
_spmd_lib_def = torch.library.Library("dummy", "DEF")
_spmd_lib_def.define("tag_grad(Tensor self) -> Tensor")

_spmd_lib_impl = torch.library.Library("dummy", "IMPL")
_spmd_lib_impl.impl("tag_grad", lambda x: x, "CompositeExplicitAutograd")


class _PyTreeCodeGenOutputsOnly(_PyTreeCodeGen):
    # pyre-ignore[3]
    def process_inputs(self, *args: Any) -> Any:
        return args

    # pyre-ignore[2, 3]
    def gen_fn_def(self, free_vars, maybe_return_annotation):
        return CodeGen.gen_fn_def(self, free_vars, maybe_return_annotation)


def _to_caller_flattened_graph_module(gm: fx.GraphModule) -> fx.GraphModule:
    """Move the responsibility of flattening the input arguments from the
    graph module to the caller.

    Example:

        output = gm(my_struct)

        gm = gm(to_caller_flattened_graph_module)

        output = gm(*pytree.flatten(my_struct)[0])
    """
    # pyre-ignore[16]
    gm._graph._codegen = _PyTreeCodeGenOutputsOnly(
        pytree_info=_PyTreeInfo(
            # pyre-ignore[6]
            orig_args=None,  # type: ignore[arg-type]
            # pyre-ignore[6]
            in_spec=None,  # type: ignore[arg-type]
            # pyre-ignore[16]
            out_spec=gm._graph._codegen.pytree_info.out_spec,
        )
    )
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm



@contextmanager
def gradients_tagging(params: Dict[str, nn.Parameter]):
    """
    This is a helper function that tags the gradient of the parameters
    with a special tag, so that we can identify them during SPMD expansion.

    It's safe to trace those hooks and we would remove those nodes later.
    """

    # tagging_hooks: List[RemovableHandle] = []
    all_red_hooks: List[RemovableHandle] = []
    try:
        for p in params.values():
            # h = p.register_hook(lambda grad: torch.ops.dummy.tag_grad(grad))
            h2 = p.register_hook( lambda grad: all_reduce(grad, reduceOp="avg", group=dist.group.WORLD))
            # tagging_hooks.append(h)
            all_red_hooks.append(h2)
        yield
    finally:
        # remove those hooks after tracing
        # for h in tagging_hooks:
        #     h.remove()
        for h in all_red_hooks:
            h.remove()


@contextmanager
def _rematerialize_optimizer(
    opt: optim.Optimizer,
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


@contextmanager
def _enable_compile():
    # The return value of torch._utils.is_compiling changes optimizer behavior.
    # We need that function to return True to include optimizer in the graph.
    # See: https://github.com/pytorch/pytorch/blob/a524123c91ab399c9dd6882c1189596dd77e7734/torch/optim/optimizer.py#L41
    def f_true():
        return True

    orig_is_compiling_code = torch._utils.is_compiling.__code__
    torch._utils.is_compiling.__code__ = f_true.__code__
    try:
        yield
    finally:
        torch._utils.is_compiling.__code__ = orig_is_compiling_code


@dataclass
class _CompiledResult:
    gm: fx.GraphModule
    mod: nn.Module
    opt: Optional[torch.optim.Optimizer]
    flat_state: List[torch.Tensor]


def _compile(func: Callable, *args: Any, **kwargs: Any):
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

    named_states: Dict[str, nn.Parameter] = {}
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

    def stateless_func(
        func: Callable,
        params: Dict[str, nn.Parameter],
        buffers: Dict[str, torch.Tensor],
        named_states: Dict[str, nn.Parameter],
        args: Any,
        kwargs: Any,
    ):
        with stateless._reparametrize_module(
            mod, {**params, **buffers}
        ), _rematerialize_optimizer(
            opt, named_states, params
        ) if opt else nullcontext():
            # Installing hooks onto gradients to identify the gradients.
            with gradients_tagging(params):
                ret = func(*args, **kwargs)

            # the return value of the function must be the original return value
            # updated paramaters and updated optimizer states
            return ret, list(mod.parameters()), list(named_states.values())

    tracing_mode = "fake"
    fake_mode = FakeTensorMode()

    def _get_fake_args(arg: torch.Tensor) -> torch.Tensor:
        fake_arg = fake_mode.from_tensor(arg)
        return fake_arg

    args = pytree.tree_map_only(torch.Tensor, _get_fake_args, args)
    kwargs = pytree.tree_map_only(torch.Tensor, _get_fake_args, kwargs)

    with _enable_compile(), torch.autograd.detect_anomaly(check_nan=False):
        gm = make_fx(
            partial(stateless_func, func),
            tracing_mode=tracing_mode,
            decomposition_table=SPMD_DECOMP_TABLE,
            _allow_non_fake_inputs=False,
        )(params, buffers, named_states, args, kwargs)

    params_and_buffers: Dict[str, Union[torch.Tensor, nn.Parameter]] = {
        **params,
        **buffers,
    }

    flat_state, _ = pytree.tree_flatten([params_and_buffers, named_states])

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.detach.default:
            input_node = node.all_input_nodes[0]
            node.replace_all_uses_with(input_node)
            if len(node.users) == 0:
                gm.graph.erase_node(node)
        if node.target == torch.ops.c10d_functional.wait_tensor.default:
            all_red_node = node.all_input_nodes[0]
            grad_node = all_red_node.all_input_nodes[0]
            while(grad_node.target == torch.ops.c10d_functional.wait_tensor.default):
                node.replace_all_uses_with(grad_node)
                if (len(node.users) == 0):
                    gm.graph.erase_node(node)
                all_red_node = grad_node.all_input_nodes[0]
                grad_node = all_red_node.all_input_nodes[0]
                
    gm = _to_caller_flattened_graph_module(gm)

    return _CompiledResult(gm, mod, opt, flat_state)


# Note that the Python convention of __dict__ requires the key to be str.
# TODO: ensure the key is unique.
COMPILED_OBJECT_KEY = "_compiled_obj"


def compile(
    gm_transformations: Optional[Callable[[fx.GraphModule], fx.GraphModule]] = None,
):
    r"""
    Compile and optimize a callable, which can be a train step within a training
    loop. This method will extract :class:`nn.Module` and :class:`torch.optim.Optimizer`
    instances from the input arguments and trace operations applied to their
    parameters and states.

    Args:
        gm_transformation (Optional[Callable[fx.GraphModule, fx.GraphModule]]):
            a callback that will be called after the original callable is
            compiled (usually after the first iteration) to
            transform the compiled GraphModule into a new optimized one.
    """

    def compile_inner(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_train_step = kwargs.pop("last_train_step", False) if kwargs else False
            first_iter = False
            # Put the COMPILED_OBJECT_KEY in ``wrapper`` instead of ``func`` as
            # ``wrapper`` is the one that users will get.
            compiled_obj = wrapper.__dict__.get(COMPILED_OBJECT_KEY, None)
            if compiled_obj is None:
                first_iter = True
                compiled_obj = _compile(func, *args, **kwargs)
                wrapper.__dict__[COMPILED_OBJECT_KEY] = compiled_obj
            graph_prof = GraphProfiler(compiled_obj.gm)
            # print(compiled_obj.gm.graph)
            flat_inps = compiled_obj.flat_state + pytree.tree_flatten([args, kwargs])[0]
            with torch.no_grad():
                # N.B.: we don't need autograd as backward has already been
                # captured in the graph.
                if first_iter and gm_transformations:
                    # print(compiled_obj.gm.graph)
                    compiled_obj.gm = gm_transformations(compiled_obj.gm)
                if not last_train_step:
                    output = compiled_obj.gm(*flat_inps)[0]
                else:
                    # This is the last train step. Call IterGraphModule.forward()
                    # with the `last_iter` argument and catch the exception in
                    # case the compiled_obj is not wrapped with IterGraphModule.
                    try:
                        output = compiled_obj.gm(*flat_inps, last_iter=last_train_step)[
                            0
                        ]
                    except TypeError as e:
                        if "last_iter" not in str(e):
                            raise e
                        output = compiled_obj.gm(*flat_inps)[0]

                return output

        return wrapper

    return compile_inner
