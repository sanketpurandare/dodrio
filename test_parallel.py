# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch.distributed._spmd.api import compile
from torch.distributed._spmd.parallel_mode import DataParallel
from torch.distributed._tensor import Replicate
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


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


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.net1 = nn.Linear(50, 32)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(32, 8)

    def forward(self, x):
        return SEPFunction.apply(torch.sigmoid(self.net2(self.relu(self.net1(x)))))

    def reset_parameters(self, *args, **kwargs):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


# simple train step definition, just an example
def train_step(model, optim, train_batch):
    def loss_fn(out, labels):
        return (out - labels).sum()

    optim.zero_grad()
    inputs, labels = train_batch

    out = model(inputs)
    loss = loss_fn(out, labels)
    loss.backward()
    optim.step()
    return loss


class TestDataParallel(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_data_parallel(
        self,
        mod,
        ddp_mod,
        opt,
        ddp_opt,
        inp,
        train_step,
        data_parallel_mode,
        data_parallel_options=None,
    ):
        ddp_inp = deepcopy(inp)

        # need one step to warm up optimizers
        train_step(mod, opt, inp)
        opt.zero_grad()
        # DDP run full train step once to align with the warmup
        train_step(ddp_mod, ddp_opt, ddp_inp)
        ddp_opt.zero_grad()

        # train a DDP model once manually as DDP grads are different
        torch.sum(ddp_mod(ddp_inp[0]) - ddp_inp[1]).backward()
        ddp_opt.step()

        # compile it with replicate and run step once
        compiled_fn = compile(
            parallel_mode=DataParallel(
                parallel_style=data_parallel_mode, _preserve_node_type=True
            )
        )(train_step)
        compiled_fn(mod, opt, inp)

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            # mod parameters are DTensors, convert to local tensor before compare
            if data_parallel_mode == "fully_shard":
                # gather the shards for comparison
                p1_replica = p1.redistribute(placements=[Replicate()])
                p1_local_param = p1_replica.to_local()
            else:
                p1_local_param = p1.to_local()
            self.assertEqual(p1_local_param, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_sgd(self):
        sgd_configs = [
            {"lr": 0.1},
            {"lr": 0.1, "momentum": 0.9},
            {"lr": 0.1, "momentum": 0.9, "foreach": True},
        ]

        for config in sgd_configs:
            mod = SimpleMLP().cuda(self.rank)
            opt = torch.optim.SGD(mod.parameters(), **config)

            train_batch = (
                torch.randn((128, 50), device=torch.device(self.rank)),
                torch.randn((128, 8), device=torch.device(self.rank)),
            )

            ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
            ddp_opt = torch.optim.SGD(ddp_mod.parameters(), **config)

            self._test_data_parallel(
                mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "replicate"
            )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicate_adam_fused(self):
        mod = SimpleMLP().cuda(self.rank)
        opt = torch.optim.Adam(mod.parameters(), lr=0.1, fused=True)

        train_batch = (
            torch.randn((128, 50), device=torch.device(self.rank)),
            torch.randn((128, 8), device=torch.device(self.rank)),
        )

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.Adam(ddp_mod.parameters(), lr=0.1, fused=True)

        self._test_data_parallel(
            mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "replicate"
        )

    # @skip_if_lt_x_gpu(2)
    # @with_comms
    # def test_fully_shard_sgd(self):
    #     mod = SimpleMLP().cuda(self.rank)
    #     opt = torch.optim.SGD(mod.parameters(), lr=0.1)

    #     train_batch = (
    #         torch.randn(128, 50).to(self.rank),
    #         torch.randn(128, 8).to(self.rank),
    #     )

    #     ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
    #     ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.1)
    #     self._test_data_parallel(
    #         mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "fully_shard"
    #     )

    # @skip_if_lt_x_gpu(2)
    # @with_comms
    # def test_fully_shard_sgd_momentum(self):
    #     mod = SimpleMLP().cuda(self.rank)
    #     opt = torch.optim.SGD(mod.parameters(), lr=0.1, momentum=0.9)

    #     train_batch = (
    #         torch.randn(128, 50).to(self.rank),
    #         torch.randn(128, 8).to(self.rank),
    #     )

    #     ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
    #     ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.1, momentum=0.9)
    #     self._test_data_parallel(
    #         mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "fully_shard"
    #     )

    # @skip_if_lt_x_gpu(2)
    # @with_comms
    # def test_fully_shard_sgd_foreach(self):
    #     mod = SimpleMLP().cuda(self.rank)
    #     opt = torch.optim.SGD(mod.parameters(), lr=0.1, momentum=0.9, foreach=True)

    #     train_batch = (
    #         torch.randn(128, 50).to(self.rank),
    #         torch.randn(128, 8).to(self.rank),
    #     )

    #     ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
    #     ddp_opt = torch.optim.SGD(
    #         ddp_mod.parameters(), lr=0.1, momentum=0.9, foreach=True
    #     )

    #     self._test_data_parallel(
    #         mod, ddp_mod, opt, ddp_opt, train_batch, train_step, "fully_shard"
    #     )


if __name__ == "__main__":
    if False:
        run_tests()
