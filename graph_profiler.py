# type: ignore
# pyre-ignore-all-errors
import logging
import os
import pdb
from dataclasses import fields
from statistics import mean
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, cast

import dill as pickle
import torch
import torch.distributed as dist
import torch.nn as nn
from functorch.compile import aot_module, make_boxed_func
from torch import fx
from torch.autograd.profiler_util import EventList
from torch.distributed._spmd.data_parallel import NodeType
from torch.fx.node import map_arg
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from graph_profiler_utils import (BiDict, IntermediateNodeInfo, NodeInfo,
                                  ProfileMode, ProfInfo, TensorStatus,
                                  get_tensor_stats)
from graph_utils import OP

MEM_LIMIT = 0
PROF_DIR = "./"

if torch.cuda.is_available():
    MEM_LIMIT = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory


class GraphProfiler(fx.Interpreter):
    r"""The main GraphProfiler class that extends the fx.Interpreter and runs
    the input graph module node by node, collecting profiling information for
    each one of them.
    Args:
    gm (fx.GraphModule): The fx graphmodule to initialize the
                                GraphProfiler.
    sync (bool): Flag to indicate whether the cuda stream should be
                synchronized for each node operation.
    profile_mode (str): The Graph Profiler provides three profiling
                        modes,``default``, ``memory`` and ``swap``.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        sync: bool = False,
        profile_mode: str = "default",
    ):
        super().__init__(gm, True)
        self.gm = gm
        logging.info(f"Current Device:  {torch.cuda.current_device()}")
        torch.cuda.reset_peak_memory_stats()

        logging.info("Initializing Graph Profiler")
        self.prefix_str = "iter_graph_profiler"
        self.sync = sync
        self.node_info: Dict[fx.Node, NodeInfo] = {}
        self.profile_mode = profile_mode

        self.total_runtime_sec: List[float] = []
        self.attr_map: Dict[fx.Node, Any] = {}
        self.node_active_mem: Dict[fx.Node, List[int]] = {}
        self.node_peak_mem: Dict[fx.Node, List[int]] = {}
        self.runtimes_sec: Dict[fx.Node, float] = {}
        self.swaptimes_sec: Dict[fx.Node, float] = {}
        self.node_cuda_time: Dict[fx.Node, float] = {}
        self.node_cpu_time: Dict[fx.Node, float] = {}
        self.node_cuda_swaptime: Dict[fx.Node, float] = {}
        self.node_cpu_swaptime: Dict[fx.Node, float] = {}
        self.intermediate_nodes: List[fx.Node] = []
        self.torch_profiler: Optional[torch.profiler.profile] = None
        self.needs_summary: bool = True
        self.param_grad_map: BiDict[fx.Node, fx.Node] = BiDict()
        self.params: List[fx.Node] = []
        self.grads: List[fx.Node] = []
        self.env = {}

        # Can define any variables that you need to measure the runtime events
        # at the Node level
        self._init_node_info()

    def _init_node_info(self) -> None:
        # print(self.gm.graph)
        # Assign ranks to nodes according to the graph topological order and
        # create a NodeInfo object for each node.
        rank = 0
        for node in self.gm.graph.nodes:
            node: fx.Node = node
            n_info = NodeInfo()
            n_info.rank = rank
            rank += 1
            self.node_info[node] = n_info

            # Find the forward end and backward start dummy nodes
            if node.name == "sep" and node.target == torch.ops.separator.sep.default:
                self.forward_end = node
            elif (
                node.name == "sep_backward"
                and node.target == torch.ops.separator.sep_backward.default
            ):
                self.backward_start = node

            # if node.target == torch.ops.c10d_functional.all_reduce.default:
            #     input_node = node.all_input_nodes[0]
            #     self.node_info[input_node].node_type = NodeType.GRAD

            if node.target == torch.ops.aten._fused_adam.default:
                param_adam_args = node.args[0]
                wait_adam_args = node.args[1]

                assert len(param_adam_args) == len(
                    wait_adam_args
                ), "The length of params and grads should be the same"
                grad_adam_args = []

                for wait_node in wait_adam_args:
                    assert isinstance(
                        wait_node, fx.Node
                    ), "Expected wait to be an fx.Node instance"
                    assert (
                        wait_node.target
                        == torch.ops.c10d_functional.wait_tensor.default
                    ), "Should have been a wait node"
                    all_red_node = wait_node.all_input_nodes[0]
                    grad = all_red_node.all_input_nodes[0]
                    self.node_info[grad].node_type = NodeType.GRAD
                    grad_adam_args.append(grad)

                for param in param_adam_args:
                    assert isinstance(
                        param, fx.Node
                    ), "Expected param to be an fx.Node instance"
                    assert (
                        param.op == OP.PLACEHOLDER
                    ), "Expected all params nodes to be of type PLACEHOLDER"
                    self.node_info[param].node_type = NodeType.PARAM

                for param, grad in zip(param_adam_args, grad_adam_args):
                    self.param_grad_map[param] = grad

        logging.info(
            f"Forward End: {self.forward_end.name} Rank: {self.node_info[self.forward_end].rank}"
        )
        logging.info(
            f"Backward Start: {self.backward_start.name} Rank: {self.node_info[self.backward_start].rank}"
        )

        # We define intermediate nodes as the nodes that are generated during forward pass
        # and also used in the backward pass

        for node in self.gm.graph.nodes:
            if (
                node.op != OP.PLACEHOLDER
                and self.node_info[node].rank < self.node_info[self.forward_end].rank
            ):
                users = node.users
                # from the users we get the last forward use
                # and the first backward use using ranks
                last_forward = None
                first_backward = None
                for user in users:
                    u_info = self.node_info[user]
                    if u_info.rank < self.node_info[self.forward_end].rank:
                        if last_forward is None:
                            last_forward = user
                        elif self.node_info[last_forward].rank < u_info.rank:
                            last_forward = user
                    if u_info.rank > self.node_info[self.backward_start].rank:
                        if first_backward is None:
                            first_backward = user
                        elif self.node_info[first_backward].rank > u_info.rank:
                            first_backward = user
                if last_forward is not None and first_backward is not None:
                    n_info = self.node_info[node]
                    self.node_info[node] = IntermediateNodeInfo(n_info)
                    n_info = None
                    self.intermediate_nodes.append(node)
                    self.node_info[node].node_type = NodeType.ACT
                    self.node_info[last_forward].last_forward_uses.append(node)
                    self.node_info[first_backward].first_back_uses.append(node)
                    self.node_info[node].first_back_access = first_backward
                    self.node_info[node].last_forward_access = last_forward
                    logging.info(
                        f"Intermediate Node: {node.name} First Backward: {first_backward.name} Last Forward: {last_forward.name}"
                    )

            elif (
                self.node_info[node].node_type == NodeType.PARAM
                and node.op == OP.PLACEHOLDER
            ):
                users = node.users
                # from users we get the first use of the parameter in the forward pass
                first_forward = None
                for user in users:
                    u_info = self.node_info[user]
                    if u_info.rank < self.node_info[self.forward_end].rank:
                        if first_forward is None:
                            first_forward = user
                        elif self.node_info[first_forward].rank > u_info.rank:
                            first_forward = user
                if first_forward is not None:
                    self.node_info[node].first_forward_access = first_forward

    def meta_run(self, *args) -> Any:
        args_iter = iter(args)
        for n in self.module.graph.nodes:
            if n.op == OP.PLACEHOLDER:
                self.env[n] = next(args_iter, None)
        args = None
        return self.run([])

    def run(self, *args) -> Any:
        self.static_memory: int = torch.cuda.memory_allocated()
        return_val = super().run(*args, initial_env=self.env)
        args = None
        self.env = {}
        return return_val

    def _swap_out_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be offloaded
        # 2) Retrieve their CPU reference (if none allocate a CPU tensor in
        #    pinned memory)
        # 3) Copy the tensor to the CPU, add the CPU tensor to the Interpreter
        #    environment
        # 4) Delete the GPU tensor
        nodes_to_offload = self.node_info[node].last_forward_uses
        for o_node in nodes_to_offload:
            o_info = cast(IntermediateNodeInfo, self.node_info[o_node])
            cpu_ref: torch.Tensor = o_info.cpu_ref
            tensor = self.env[o_node]
            assert isinstance(tensor, torch.Tensor)
            if cpu_ref is None:
                cpu_ref = torch.zeros(
                    tensor.size(), dtype=tensor.dtype, layout=tensor.layout
                ).pin_memory()
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {o_node.name}"
            logging.info(f"Swapping Out: {o_node.name}")
            with record_function(f"{self.prefix_str}_{o_node.name}_swap"):
                cpu_ref = cpu_ref.copy_(tensor, False)
            if self.sync:
                torch.cuda.synchronize()
            o_info.status = TensorStatus.cpu
            o_info.cpu_ref = cpu_ref
            self.env[o_node] = cpu_ref
            del tensor
            tensor = None
            cpu_ref = None

    def _swap_in_node(self, node: fx.Node) -> None:
        # 1) Get the nodes to be prefetched
        # 2) Retrieve their CPU reference (assert if it resides in pinned
        #    memory)
        # 3) Copy the tensor to GPU memory and add it to the Interpreter
        #    environment
        # 4) Update the state of intermediate tensor in NodeInfo
        nodes_to_fetch = self.node_info[node].first_back_uses
        for p_node in nodes_to_fetch:
            p_info = cast(IntermediateNodeInfo, self.node_info[p_node])
            p_info.status = TensorStatus.gpu
            cpu_ref = cast(torch.Tensor, p_info.cpu_ref)
            # assert isinstance(cpu_ref, torch.Tensor), f"CPU ref is not a tensor for {p_node.name}"
            assert cpu_ref.is_pinned, f"CPU ref is not pinned for {p_node.name}"
            logging.info(f"Swapping In: {p_node.name}")
            with record_function(f"{self.prefix_str}_{p_node.name}_swap"):
                tensor = cpu_ref.to(
                    device=torch.cuda.current_device(),
                    memory_format=torch.preserve_format,
                    non_blocking=False,
                )
            self.env[p_node] = tensor.contiguous()
            tensor = None
            if self.sync:
                torch.cuda.synchronize()

    def _verify_inputs_on_gpu(self, node: fx.Node):
        for i_node in node.all_input_nodes:
            in_value = self.env[i_node]
            if isinstance(in_value, torch.Tensor):
                assert (
                    in_value.get_device() == torch.cuda.current_device()
                ), f"Mismatch in device types for {i_node.name} which is the input of node {node.name}"

    def run_node(self, node: fx.Node) -> Any:
        if node.op == OP.PLACEHOLDER:
            return super().run_node(node)

        # Preftech the tensors that have been offloaded and have their first uses.
        if (
            self.profile_mode == ProfileMode.swap
            and self.node_info[node].rank > self.node_info[self.backward_start].rank
        ):
            self._swap_in_node(node)
            self._verify_inputs_on_gpu(node)

        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()

        with record_function(f"{self.prefix_str}_{node.name}"):
            return_val = super().run_node(node)
        if self.sync:
            torch.cuda.synchronize()

        if node.op == OP.GET_ATTR:
            self.attr_map[node] = return_val

        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            mem_stats = torch.cuda.memory_stats()
            self.node_peak_mem.setdefault(node, [])
            self.node_peak_mem[node].append(mem_stats["active_bytes.all.peak"])
            self.node_active_mem.setdefault(node, [])
            self.node_active_mem[node].append(mem_stats["active_bytes.all.current"])
            if node in self.intermediate_nodes:
                int_n_info = cast(IntermediateNodeInfo, self.node_info[node])
                assert isinstance(return_val, torch.Tensor)
                (
                    int_n_info.size,
                    int_n_info.numel,
                    int_n_info.memory_size,
                ) = get_tensor_stats(return_val)

        # Offload the tensors that have last uses at this node during forward pass.
        if (
            self.profile_mode == ProfileMode.swap
            and self.node_info[node].rank < self.node_info[self.forward_end].rank
        ):
            self._swap_out_node(node)

        return return_val

    def reset_stats(self) -> None:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        self.total_runtime_sec: List[float] = []
        self.node_active_mem: Dict[fx.Node, List[Any]] = {}
        self.node_peak_mem: Dict[fx.Node, List[Any]] = {}
        self.runtimes_sec: Dict[fx.Node, List[float]] = {}
        self.swaptimes_sec: Dict[fx.Node, List[float]] = {}

    def get_idle_times(self) -> None:
        for i_node in self.intermediate_nodes:
            n_info = cast(IntermediateNodeInfo, self.node_info[i_node])
            last_use = n_info.last_forward_access
            n_info.idle_time = self.total_runtime - (
                self.node_info[last_use].cumulative_run_time + n_info.swap_time
            )

            first_use = n_info.first_back_access
            n_info.idle_time += self.node_info[first_use].cumulative_run_time - (
                self.node_info[first_use].run_time + n_info.swap_time
            )

    def get_peakmem_usage(self) -> None:
        if self.profile_mode == ProfileMode.swap:
            intermediate_mem = 0
            # for i_node in self.intermediate_nodes:
            #     n_info = cast(IntermediateNodeInfo, self.node_info[i_node])
            #     intermediate_mem += n_info.memory_size

            self.peak_start = None
            self.peak_end = None
            peak_interval: bool = False
            peak_end_reset: bool = False
            self.max_peak_mem = 0
            self.min_peak_mem = 0
            for node in self.module.graph.nodes:
                if node.op == OP.PLACEHOLDER:
                    continue
                if self.node_info[node].rank > self.node_info[self.backward_start].rank:
                    nodes_to_prefetch = self.node_info[node].first_back_uses
                    if nodes_to_prefetch is not None:
                        for p_node in nodes_to_prefetch:
                            intermediate_mem -= self.node_info[p_node].memory_size
                min_peak_mem = self.node_info[node].peak_mem
                peak_mem = min_peak_mem + intermediate_mem
                if peak_mem > MEM_LIMIT:
                    peak_interval = True
                    peak_end_reset = True
                    if self.peak_start is None:
                        self.peak_start = node
                else:
                    peak_interval = False
                    if peak_end_reset:
                        self.peak_end = node
                        peak_end_reset = False

                self.node_info[node].in_peak_interval = peak_interval
                self.node_info[node].total_peak_mem = peak_mem
                self.max_peak_mem = max(self.max_peak_mem, peak_mem)
                self.min_peak_mem = max(self.min_peak_mem, min_peak_mem)

                if self.node_info[node].rank < self.node_info[self.forward_end].rank:
                    nodes_to_offload = self.node_info[node].last_forward_uses
                    if nodes_to_offload is not None:
                        for o_node in nodes_to_offload:
                            intermediate_mem += self.node_info[o_node].memory_size
        else:
            peak_mem_usages = [
                self.node_info[n].peak_mem
                for n in self.module.graph.nodes
                if n.op != OP.PLACEHOLDER
            ]
            self.max_peak_mem = max(peak_mem_usages)
            self.min_peak_mem = min(peak_mem_usages)
            self.peak_start = None
            self.peak_end = None

    def get_node_runtimes(self):
        assert self.torch_profiler is not None
        event_list_avg: EventList = self.torch_profiler.key_averages()
        event_dict: Dict[str, Tuple[float, float]] = {}
        prefix = self.prefix_str
        for e in event_list_avg:
            if prefix in e.key:
                event_dict[e.key] = (e.cuda_time, e.cpu_time)
        for n in self.module.graph.nodes:
            if n.op != OP.PLACEHOLDER:
                cuda_time, cpu_time = event_dict[f"{self.prefix_str}_{n.name}"]
                self.node_cuda_time[n] = cuda_time / 1000.0
                self.node_cpu_time[n] = cpu_time / 1000.0
                self.runtimes_sec[n] = max(cpu_time, cuda_time) / 1000.0
        if self.profile_mode == ProfileMode.swap:
            for int_n in self.intermediate_nodes:
                cuda_time, cpu_time = event_dict[f"{self.prefix_str}_{int_n.name}_swap"]
                self.node_cuda_swaptime[int_n] = cuda_time / 1000.0
                self.node_cpu_swaptime[int_n] = cpu_time / 1000.0
                self.swaptimes_sec[int_n] = max(cpu_time, cuda_time) / 1000.0

    def get_total_cumulative_run_time(self) -> None:
        self.total_runtime = 0

        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info = self.node_info[node]
            self.total_runtime += n_info.run_time
            n_info.cumulative_run_time = self.total_runtime

    def calibrate_aggregate_stats(self) -> None:
        self.get_total_cumulative_run_time()
        self.get_idle_times()
        self.get_peakmem_usage()

    def summarize(self) -> None:
        if not self.needs_summary:
            return

        self.get_node_runtimes()
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue

            n_info = self.node_info[node]
            n_info.run_time = self.runtimes_sec.get(node, 1.0)
            n_info.cuda_time = self.node_cuda_time.get(node, 1.0)
            n_info.cpu_time = self.node_cpu_time.get(node, 1.0)
            n_info.exe_time = n_info.run_time

            if self.profile_mode not in [ProfileMode.swap, ProfileMode.memory]:
                continue
            n_info.peak_mem = max(self.node_peak_mem.setdefault(node, [0]))
            n_info.active_mem = max(self.node_active_mem.setdefault(node, [0]))

            if (
                node in self.intermediate_nodes
                and self.profile_mode == ProfileMode.swap
            ):
                n_info: IntermediateNodeInfo = self.node_info[node]
                n_info.swap_time = self.swaptimes_sec[node]
        self.calibrate_aggregate_stats()

    def print_summary(self) -> str:
        try:
            import tabulate
        except ImportError:
            return "No tabulate module is found, skip printing summary."

        node_summaries: List[List[Any]] = []
        mean_total_runtime = self.total_runtime
        logging.info(f"Execution Time (ms): {self.total_runtime}")
        logging.info(f"Max Peak Mem Usage (B): {self.max_peak_mem}")

        headers: List[str] = [
            "Target",
            "Op",
            "Average runtime (ms)",
            "Pct total runtime",
            "CUDA time(ms)",
            "CPU time(ms)",
        ]
        if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
            headers.extend(
                [
                    "Mem Active (B)",
                    "Mem Peak Active(B)",
                    "Tensor Size(B)",
                ]
            )
        if self.profile_mode == ProfileMode.swap:
            print(
                "Peak Interval : ",
                str(self.peak_start),
                " - ",
                str(self.peak_end),
            )
            headers.extend(
                [
                    "Swap Time (ms)",
                    "Idle_time(ms)",
                    "Simulated Peak Active(B)",
                ]
            )
        for node in self.module.graph.nodes:
            if node.op == OP.PLACEHOLDER:
                continue
            n_info: NodeInfo = self.node_info[node]
            pct_total = n_info.run_time / mean_total_runtime * 100
            val_list = [
                node.target,
                str(node),
                n_info.run_time,
                pct_total,
                n_info.cuda_time,
                n_info.cpu_time,
            ]
            if self.profile_mode in [ProfileMode.swap, ProfileMode.memory]:
                val_list.extend([n_info.active_mem, n_info.peak_mem])
            if node in self.intermediate_nodes:
                n_info: IntermediateNodeInfo = n_info
                if self.profile_mode == ProfileMode.memory:
                    val_list.append(n_info.memory_size)
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(
                        [n_info.memory_size, n_info.swap_time, n_info.idle_time]
                    )
            else:
                if self.profile_mode == ProfileMode.memory:
                    val_list.append("")
                if self.profile_mode == ProfileMode.swap:
                    val_list.extend(["", "", ""])
            if self.profile_mode == ProfileMode.swap:
                val_list.append(n_info.total_peak_mem)
            node_summaries.append(val_list)
        return tabulate.tabulate(node_summaries, headers=headers)

    def get_prof_stats(self) -> Dict[str, ProfInfo]:
        profile_stats: Dict[str, ProfInfo] = {}
        for node, ninfo in self.node_info.items():
            if node in self.intermediate_nodes:
                pinfo = ProfInfo(
                    ninfo.run_time,
                    ninfo.cuda_time,
                    ninfo.cpu_time,
                    ninfo.peak_mem,
                    ninfo.active_mem,
                    True,
                    ninfo.swap_time,
                    ninfo.size,
                    ninfo.memory_size,
                    ninfo.numel,
                )
                profile_stats[node.name] = pinfo
            else:
                pinfo = ProfInfo(
                    ninfo.run_time,
                    ninfo.cuda_time,
                    ninfo.cpu_time,
                    ninfo.peak_mem,
                    ninfo.active_mem,
                    False,
                )
                profile_stats[node.name] = pinfo

        return profile_stats

    def set_prof_stats(self, profile_stats: Dict[str, ProfInfo]) -> None:
        for node, ninfo in self.node_info.items():
            pinfo: ProfInfo = profile_stats[node.name]
            ninfo.run_time = pinfo.run_time
            ninfo.cuda_time = pinfo.cuda_time
            ninfo.cpu_time = pinfo.cpu_time
            ninfo.active_mem = pinfo.active_mem
            ninfo.peak_mem = pinfo.peak_mem

            if node in self.intermediate_nodes:
                ninfo.swap_time = pinfo.swap_time
                ninfo.size = pinfo.size
                ninfo.memory_size = pinfo.memory_size
                ninfo.numel = pinfo.numel

    def aggregate_prof_stats(
        self, profile_stats_list: List[Dict[str, ProfInfo]]
    ) -> None:
        for node, ninfo in self.node_info.items():
            pinfos: List[ProfInfo] = [
                profile_stats[node.name] for profile_stats in profile_stats_list
            ]
            ninfo.run_time = mean([pinfo.run_time for pinfo in pinfos])
            ninfo.cuda_time = mean([pinfo.cuda_time for pinfo in pinfos])
            ninfo.cpu_time = mean([pinfo.cpu_time for pinfo in pinfos])
            ninfo.peak_mem = max([pinfo.peak_mem for pinfo in pinfos])
            ninfo.active_mem = max([pinfo.active_mem for pinfo in pinfos])

            if node in self.intermediate_nodes:
                ninfo.swap_time = mean([pinfo.swap_time for pinfo in pinfos])
                ninfo.size = max([pinfo.size for pinfo in pinfos])
                ninfo.memory_size = max([pinfo.memory_size for pinfo in pinfos])
                ninfo.numel = max([pinfo.numel for pinfo in pinfos])
        self.calibrate_aggregate_stats()

    def save_node_info(self, mod_id: str):
        dirname = f"{PROF_DIR}/{mod_id}"
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        profile_stats = self.get_prof_stats()
        filename = f"{dirname}/{mod_id}.profinfo"
        with open(filename, "wb") as outp:
            pickle.dump(profile_stats, outp, pickle.HIGHEST_PROTOCOL)

    def load_prof_info(self, mod_id):
        dirname = f"{PROF_DIR}/{mod_id}"
        filename = f"{dirname}/{mod_id}.profinfo"
        with open(filename, "rb") as inp:
            profile_stats: Dict[str, ProfInfo] = pickle.load(inp)
        self.set_prof_stats(profile_stats)
        self.calibrate_aggregate_stats()


class ProfilerEngine:
    r"""It provides an API to initialize and run the graph profiler.
    It provides the run function which takes an optional
    argument for running warm-up iterations before doing the actual profiling.
    Args:
    gm(fx.GraphModule): The IterGraphModule to be profiled.
    warm_up_iters(int): Number of warmup iterations to run.
    profile_iters(int): Number of iterations to run the profiler.
    profile_mode (str): The Graph Profiler provides three profiling modes,
                        ``default``,``memory`` and ``swap``.
                default: Measure the per node run-time, marks the intermediate
                        nodes (activations saved from forward pass and needed in
                        backward pass), registers their last use in the forward
                        pass and first use in the backward pass, measures this
                        idle time and, marks the irst use of the model parameter
                        nodes in the forward pass.
                memory: All of the above plus active memory usage,
                        peak memory usage and intermediate (activation) memory.
                swap:   All the of the above plus profiles in a low memory
                        mode, pushing all of the activations to the CPU
                        memory during the forward pass and fetches them
                        back when they are needed in the backward pass.
                        It measures the time to swap each of the intermediate
                        tensors (activations) to CPU memory, back and forth.
                        Allows profiling graphs way larger than GPU memory.
    """

    def __init__(
        self,
        gm: fx.GraphModule,
        warm_up_iters: int = 0,
        profile_iters: int = 1,
        profile_mode: str = "default",
    ) -> None:
        self.gm = gm
        self.warm_up_iters = warm_up_iters
        self.profile_iters = profile_iters
        self.profile_mode = profile_mode
        self.graph_profiler = GraphProfiler(self.gm, False, self.profile_mode)

    def run(self, *args, **kwargs) -> None:
        r"""
        Calls the _compile method to initialize the profiler context. Runs
        optional warm-up profiling iterations. This is sometimes essential to
        warm-up the cuda caching allocator and initilize the pinned CPU memory
        when profiling for swapping times as well. Subsequent to warm-up, all
        the profiler statistics are reset and the actual profiling is done for
        number of iterations specified.
        """
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=schedule(
                skip_first=1,
                wait=1,
                warmup=self.warm_up_iters,
                active=self.profile_iters,
            ),
        ) as torch_prof, torch.no_grad():
            for i in range(2 + self.warm_up_iters + self.profile_iters):
                if i == 0:
                    self.graph_profiler.torch_profiler = torch_prof
                if i == 2 + self.warm_up_iters:
                    self.reset_stats()
                self.graph_profiler.meta_run(*args, **kwargs)
                torch_prof.step()

    def _all_gather_node_info(self) -> None:
        """
        We gather and average the node_info across all ranks.
        The current assumption is that all ranks will have exactly the same graphs.
        The design won't work if the assumption does not hold.
        The key is we need to map the name to its
        corresponding fx.Node.
        This API is needed as different profiling may procude different
        bucketing and scheduling. As a result, without the synchronization, the
        forward and backward passes may be stuck due to different ranks hold
        different optimized graphs.
        """
        prof_stats = self.graph_profiler.get_prof_stats()
        object_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(object_list, prof_stats)
        self.graph_profiler.aggregate_prof_stats(object_list)

    def summarize(self, to_aggregate: bool = False, to_print: bool = False) -> None:
        r"""
        Aggregates all the statistics accumulated during the profiling
        iterations and makes them ready for printing.
        """
        self.graph_profiler.summarize()
        if to_aggregate:
            self._all_gather_node_info()
        if to_print:
            logging.info("\n")
            logging.info(self.graph_profiler.print_summary())

    def reset_stats(self):
        r"""
        Resets all the accumulated profiling statistics. Usualy called after
        warm-up iterations or before beginning a new profiling session.
        """
        self.graph_profiler.reset_stats()
