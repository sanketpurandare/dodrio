import importlib
import logging
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.fx.experimental.proxy_tensor import make_fx
from torchbenchmark.models import (hf_Bert, hf_GPT2, hf_GPT2_large, hf_T5,
                                   hf_T5_large, timm_vision_transformer_large)
from torchbenchmark.util.model import BenchmarkModel

from graph_compiler import compile
from graph_compiler_utils import SEPFunction

actual_model_names: List[str] = [
    "hf_Bert",
    "hf_T5",
    "hf_GPT2",
    "hf_T5_large",
    "hf_GPT2_large",
    "timm_vision_transformer_large",
]

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "torchbenchmark.models.hf_GPT2_large.Model",
    "torchbenchmark.models.hf_T5_large.Model",
    "torchbenchmark.models.timm_vision_transformer_large.Model",
    "torchbenchmark.models.hf_GPT2.Model",
    "torchbenchmark.models.hf_T5.Model",
]
model_batch_sizes: Dict[str, int] = {
    "torchbenchmark.models.hf_Bert.Model": 32,
    "torchbenchmark.models.hf_GPT2_large.Model": 4,
    "torchbenchmark.models.hf_T5_large.Model": 4,
    "torchbenchmark.models.timm_vision_transformer_large.Model": 16,
    "torchbenchmark.models.hf_GPT2.Model": 24,
    "torchbenchmark.models.hf_T5.Model": 12,
}

# class WrappedDummyModel(nn.Module):
#     def __init__(self, mod: nn.Module):
#         super().__init__()
#         self.mod = mod

#     def forward(self, *args, **kwargs):
#         return SEPFunction.apply(self.mod(*args, **kwargs))


class Experiment:
    def __init__(self, model_name: str, batch_size: int, extra_args=[]):
        pos = model_name.rfind(".")
        module = importlib.import_module(model_name[:pos])
        model_class = getattr(module, model_name[(pos + 1) :])

        model: BenchmarkModel = model_class(
            "train", "cuda", batch_size=batch_size, extra_args=extra_args
        )
        self.model = model.model
        self.model_type = type(model)

        self.batch_size = batch_size
        self.example_inputs = model.example_inputs

        if self.model_type in (
            hf_T5.Model,
            hf_GPT2.Model,
            hf_Bert.Model,
            hf_T5_large.Model,
            hf_GPT2_large.Model,
        ):

            def bert_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                loss = model(**example_inputs).loss
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = bert_train_step
            self.optimizer = model.optimizer

        elif self.model_type == timm_vision_transformer_large.Model:
            self.loss_fn = model.cfg.loss
            self._gen_target = model._gen_target

            def timm_vit_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                output = model(example_inputs)
                target = self._gen_target(output.shape[0])
                loss = self.loss_fn(output, target)
                loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.optimizer = model.cfg.optimizer
            self.train_step = timm_vit_train_step

    def init_optimizer_states(self):
        for param in self.model.parameters():
            param.grad = torch.rand_like(param)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run(self):
        self.train_step(self.model, self.optimizer, self.example_inputs)
        print("Successful.")


def run_worker(rank, world_size):
    logging.getLogger().setLevel(logging.DEBUG if rank == 0 else logging.CRITICAL)
    # logging.getLogger().setLevel(logging.DEBUG)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    logging.info(f"Number of visible devices:  {torch.cuda.device_count()}")
    torch.cuda.set_device(rank)
    torch.manual_seed(20)
    logging.critical(f"Cuda device: {torch.cuda.current_device()}")

    exp = Experiment(model_names[3], model_batch_sizes[model_names[3]])
    exp.init_optimizer_states()
    compiled_fn = compile()(exp.train_step)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

    # compiled_gm = make_fx(
    #     exp.train_step, tracing_mode="fake", _allow_non_fake_inputs=True
    # )(exp.model, exp.optimizer, exp.example_inputs)
    # print(compiled_gm.graph)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
