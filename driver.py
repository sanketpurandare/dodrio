import importlib
import logging
import os
from typing import Any, Dict, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from graph_compiler import compile
from graph_compiler_utils import SEPFunction
from torch.fx.experimental.proxy_tensor import make_fx
from torchbenchmark.models import (
    gemma_2b,
    hf_Bert,
    hf_GPT2,
    hf_GPT2_large,
    hf_T5,
    hf_T5_large,
    moondream2,
    open_llama_3b,
    timm_vision_transformer_large,
    tinyllama,
    tinyllava,
    torch_multimodal_clip,
)
from torchbenchmark.util.model import BenchmarkModel
from torch.nn.attention import SDPBackend, sdpa_kernel
torch.backends.cuda.enable_flash_sdp(enabled=True)
actual_model_names: List[str] = [
    "hf_Bert",
    "hf_T5",
    "hf_GPT2",
    "hf_T5_large",
    "hf_GPT2_large",
    "timm_vision_transformer_large",
    "torch_multimodal_clip",
    "tinyllama",
    "tinyllava",
    "gemma_2b",
    "open_llama_3b",
    "moondream2",
]

model_names: List[str] = [
    "torchbenchmark.models.hf_Bert.Model",
    "torchbenchmark.models.hf_GPT2_large.Model",
    "torchbenchmark.models.hf_T5_large.Model",
    "torchbenchmark.models.timm_vision_transformer_large.Model",
    "torchbenchmark.models.hf_GPT2.Model",
    "torchbenchmark.models.hf_T5.Model",
    "torchbenchmark.models.tinyllama.Model",
    "torchbenchmark.models.tinyllava.Model",
    "torchbenchmark.models.gemma_2b.Model",
    "torchbenchmark.models.open_llama_3b.Model",
    "torchbenchmark.models.moondream2.Model",
    "torchbenchmark.models.torch_multimodal_clip.Model",
]
model_batch_sizes: Dict[str, int] = {
    "torchbenchmark.models.hf_Bert.Model": 32,
    "torchbenchmark.models.hf_GPT2_large.Model": 4,
    "torchbenchmark.models.hf_T5_large.Model": 4,
    "torchbenchmark.models.timm_vision_transformer_large.Model": 16,
    "torchbenchmark.models.hf_GPT2.Model": 24,
    "torchbenchmark.models.hf_T5.Model": 12,
    "torchbenchmark.models.tinyllama.Model": 12,
    "torchbenchmark.models.tinyllava.Model": 12,
    "torchbenchmark.models.gemma_2b.Model": 4,
    "torchbenchmark.models.open_llama_3b.Model": 4,
    "torchbenchmark.models.moondream2.Model": 8,
    "torchbenchmark.models.torch_multimodal_clip.Model": 32,
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
        # print(model.__dict__)
        # print(self.example_inputs)
        param_count = 0
        param_tensor_count = 0
        for param in self.model.parameters():
            if not param.requires_grad:
                print("frozen param")
            param_count += param.numel()
            param_tensor_count += 1

        print(f"Model has {param_count} parameters.")
        print(f"Model has {param_tensor_count} parameter tensors.")
        print(f"Parameter Memory: {torch.cuda.memory_allocated() / 2**30} GiB")

        if self.model_type in (
            hf_T5.Model,
            hf_GPT2.Model,
            hf_Bert.Model,
            hf_T5_large.Model,
            hf_GPT2_large.Model,
            tinyllama.Model,
            tinyllava.Model,
            gemma_2b.Model,
            open_llama_3b.Model,
            moondream2.Model,
        ):

            def hf_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model(**example_inputs).loss
                        loss = SEPFunction.apply(loss)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()

            self.model.train()
            self.train_step = hf_train_step
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

        elif self.model_type == torch_multimodal_clip.Model:
            self.optimizer = model.optimizer
            self.loss_fn = model.loss_fn
            self.model.train()

            def clip_train_step(
                model: nn.Module, optim: optim.Optimizer, example_inputs: Any
            ):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    image_embedding, text_embedding = self.model(*example_inputs)
                    loss = self.loss_fn(image_embedding, text_embedding)
                    loss = SEPFunction.apply(loss)
                loss.backward()
                optim.step()
                optim.zero_grad()

            self.train_step = clip_train_step

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

    exp = Experiment(model_names[7], model_batch_sizes[model_names[7]])
    exp.init_optimizer_states()
    compiled_fn = compile()(exp.train_step)
    compiled_fn(exp.model, exp.optimizer, exp.example_inputs)

    # compiled_gm = make_fx(
    #     exp.train_step, tracing_mode="fake", _allow_non_fake_inputs=True
    # )(exp.model, exp.optimizer, exp.example_inputs)
    # print(compiled_gm.graph)


if __name__ == "__main__":
    exp = Experiment(model_names[7], model_batch_sizes[model_names[7]])
    exp.init_optimizer_states()
    torch.cuda.synchronize()
    print(f"Memory: {torch.cuda.memory_allocated() / 2**30} GiB")
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    for i in range(5):
        start_events[i].record()
        exp.run()
        end_events[i].record()
    torch.cuda.synchronize()
    iter_time = (
        sum(start_events[i].elapsed_time(end_events[i]) for i in range(2, 5)) / 3
    )
    print(f"Iter time: {iter_time} ms")
    print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 2**30} GiB")
    print(f"Peak Memory Reserved: {torch.cuda.max_memory_reserved() / 2**30} GiB")
    exit()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
