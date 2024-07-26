import logging
import os
from copy import deepcopy
from functools import wraps

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from graph_compiler import compile
from torch.distributed._functional_collectives import all_reduce
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.parallel import DistributedDataParallel as DDP


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend([nn.Linear(dim, dim), nn.ReLU()])
        self.mod = nn.Sequential(*modules)

    def forward(self, x):
        return self.mod(x)


def train_step(
    model: torch.nn.Module, optim: torch.optim.Optimizer, batch: torch.Tensor
):
    out: torch.Tensor = model(batch)
    out.sum().backward()
    optim.step()
    optim.zero_grad()


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    # logging.getLogger().setLevel(logging.DEBUG)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Number of visible devices:  {torch.cuda.device_count()}")
    torch.cuda.set_device(rank)
    torch.manual_seed(20)
    batch_size = 100
    layers = 10
    dim = 100
    num_iters = 5
    model = DummyModel(dim=dim, layers=layers).cuda()
    batch = torch.randn(batch_size, dim).cuda()
    optim = torch.optim.Adam(
        model.parameters(), lr=0.01, foreach=False, fused=True, capturable=True
    )

    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(call_all_reduce)

    # ddp_model = DDP(deepcopy(model), device_ids=[rank])
    # ddp_optim = torch.optim.Adam(
    #         ddp_model.parameters(),
    #         lr=0.01,
    #         foreach=False,
    #         fused=True,
    #         capturable=True,
    #     )
    compiled_fn = compile()(train_step)
    compiled_fn(model, optim, batch)

    # gm = make_fx(train_step)(model, optim, batch)
    # print(gm.graph)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)


def call_all_reduce(grad: torch.Tensor):
    print("This was called.")
    return all_reduce(grad, reduceOp="avg", group=dist.group.WORLD)
