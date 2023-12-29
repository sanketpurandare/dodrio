from typing import Any, Dict, List, OrderedDict, Set, Tuple

import torch.fx as fx
from graph_profiling.graph_profiler_utils import IntNodeInfo, MEM_LIMIT, NodeInfo
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.distributed._spmd.graph_profiler import GraphProfiler
from torch.distributed._spmd.graph_utils import rebuild_graph

from .graph_utils import get_output, OP, replace_subsequent_uses_of

####################################################################################
# Capuchin Algorithm greedily chooses activations to recompute based on the
# recompute ratio metric which is activation_size/compute_time.
# Higher ratio imples that we can recompute activations with lower overhead.

class Capuchin:
    def __init__(
        self,
        gm: fx.GraphModule,
        graph_profiler: GraphProfiler,
    ) -> fx.GraphModule:
        self.gm = gm
        self.graph_profiler = graph_profiler
        self.recomps: Set[fx.Node] = set()
        self._populate_prof_info()

    def _populate_prof_info(self):

        self.node_info = self.graph_profiler.node_info
        self.intermediate_nodes = self.graph_profiler.intermediate_nodes
        self.peak_end = self.graph_profiler.peak_end
        self.max_peak_mem = self.graph_profiler.max_peak_mem
        self.min_peak_mem = self.graph_profiler.min_peak_mem
        self.static_memory = self.graph_profiler.static_memory
        print("Maximum Peak Memory Requirements: ", self.max_peak_mem)
        print("Minimum Peak Memory Requirements: ", self.min_peak_mem)
        print("GPU Memory Limit: ", MEM_LIMIT)

    ##################################################################################
    # Functions for Recompute

    def _get_placeholders(self) -> List[fx.Node]:
        placeholders: List[fx.Node] = []
        for node in self.fw_module.graph.nodes:
            if node.op == "placeholder":
                placeholders.append(node)
        return placeholders

    def _update_existing_recomps(self, t: fx.Node) -> int:
        # Checks if the activation currently chosen to be recomputed (t) is one of the
        # sources  of activations already chosen to be recomputed (recomps). If yes,
        # in rp.srcs replace t with t.srcs, count the number of times this is done.
        exe_count = 1
        for rp in self.recomps:
            rp_info: IntNodeInfo = self.node_info[rp]
            t_info: IntNodeInfo = self.node_info[t]
            if t in rp_info.rcomp_activation_sources:
                rp_info.rcomp_intermediates.append(t)
                rp_info.rcomp_activation_sources = [
                    src for src in rp_info.rcomp_activation_sources if src != t
                ]
                rp_info.rcomp_activation_sources.extend(t_info.rcomp_activation_sources)
                rp_info.rcomp_param_sources.extend(t_info.rcomp_param_sources)
                exe_count += 1
        return exe_count

    def _update_rem_candidates(
        self, t: fx.Node, exe_count: int, candidates: List[fx.Node]
    ) -> None:
        t_info: IntNodeInfo = self.node_info[t]
        for cand in candidates:
            cand_info: IntNodeInfo = self.node_info[cand]
            # Case 1:
            # Checks if the activation currently chosen to be recomputed (t) is one of the
            # sources of potential candidates (cand). If yes, in cand.srcs replace t with t.srcs.
            # Count the number of times (r_count) cand is used as a source for the existing
            # recomps and multiply its recomputation time with r_count to update its exe_time.
            if t in cand_info.rcomp_activation_sources:
                cand_info.rcomp_intermediates.append(t)
                cand_info.rcomp_activation_sources = [
                    src for src in cand_info.rcomp_activation_sources if src != t
                ]
                cand_info.rcomp_activation_sources.extend(
                    t_info.rcomp_activation_sources
                )
                cand_info.rcomp_param_sources.extend(t_info.rcomp_param_sources)
                cand_info.rcomp_time += t_info.rcomp_time
                r_count = 1
                for rp in self.recomps:
                    rp_info: IntNodeInfo = self.node_info[rp]
                    if cand in rp_info.rcomp_activation_sources:
                        r_count += 1
                cand_info.exe_time = r_count * cand_info.rcomp_time
                cand_info.updateMSPS()
            # Case 2:
            # Alternatively, if the cand is one of the sources of t, update its exe time as
            # number of times t is recomputed (exe_count) with the recomputation time of cand.
            if cand in t_info.rcomp_activation_sources:
                cand_info.exe_time = exe_count * cand_info.rcomp_time
                cand_info.updateMSPS()

    def _optimize_recomps(self) -> None:
        # 1) Find the activations that no other activation recomputes in its lineage
        # 2) Find the earliest recomputable activation and add the current activation to it's output set
        top_recomps: List[fx.Node] = []
        recomputed_intermediates:List[fx.Node] = []

        for rp in self.recomps:
            recomputed_intermediates.extend(self.node_info[rp].rcomp_intermediates)
        recomputed_intermediates = set(recomputed_intermediates)

        for rp in self.recomps:
            if rp not in recomputed_intermediates:
                top_recomps.append(rp)

        rem_recomps = self.recomps - top_recomps

        for r in rem_recomps:
            ancestors:List[fx.Node] = []
            for ac in top_recomps:
                ac_info:IntNodeInfo = self.node_info[ac]
                if r in ac_info.rcomp_intermediates:
                    ancestors.add(ac)
            list.sort(ancestors, key = lambda n: self.node_info[self.node_info[n].first_back_access].rank)
            top_ancestor = ancestors.pop(0)
            assert(self.node_info[self.node_info[top_ancestor].first_back_access].rank < self.node_info[self.node_info[r].first_back_access].rank)
            anc_info: IntNodeInfo = self.node_info[top_ancestor]
            anc_info.rcomp_outs.append(r)

            for ac in ancestors:
                ac_info = self.node_info[ac]
                ac_info.rcomp_activation_sources.append(r)

        self.req_recomps = top_recomps


    def _rewrite_graph(self) -> None:
        remap:Dict[str, fx.Node] = {}
        for node in self.gm.graph.nodes:
            remap[node.name] = node
        list.sort(self.req_recomps, key = lambda n: self.node_info[self.node_info[n].first_back_access].rank)
        for rp in self.req_recomps:
            rp_info: IntNodeInfo = self.node_info[rp]
            rcomp_graph = _extract_graph_with_inputs_outputs(
                self.gm.graph,
                rp_info.rcomp_param_sources + rp_info.rcomp_activation_sources,
                rp_info.rcomp_outs,
            )
            print("Recomputation Graph for: ", str(rp))
            print(rcomp_graph)
            output = get_output(rcomp_graph)
            output_args = output.all_input_nodes
            first_back_access:fx.Node = self.node_info[rp].first_back_access
            with self.gm.graph.inserting_before(first_back_access):
                for n in rcomp_graph.nodes:
                    if n.op == OP.PLACEHOLDER or n.op == OP.OUTPUT:
                        continue
                    else:
                        new_node = self.gm.graph.node_copy(n, arg_transform= lambda x : remap[x.name])
                        if n in output_args:
                            new_intermediate_node = new_node
                            old_intermediate_node = remap[n.name]
                            replace_subsequent_uses_of(self.gm.graph, old_intermediate_node, new_intermediate_node)
                        remap[n.name] = new_node
        rebuild_graph(self.gm)

    def _prep_recomps(self):
        # for each recomp_node in recomps
        # 1) extract subgraph from the forward pass
        # 2) add the recomp_node to be deleted during it's last forward access
        # 3) add the recomp node to be recomputed during it's first backward access

        for rp in self.recomps:
            rp_info: IntNodeInfo = self.node_info[rp]
            # rp_info.rcomp_param_sources.reverse()
            # rp_info.rcomp_activation_sources.reverse()
            # rp_info.rcomp_other_sources.reverse()
            rp_info.rcomp_param_sources = list(
                OrderedDict.fromkeys(rp_info.rcomp_param_sources)
            )
            rp_info.rcomp_activation_sources = list(
                OrderedDict.fromkeys(rp_info.rcomp_activation_sources)
            )
            rp_info.rcomp_other_sources = list(
                OrderedDict.fromkeys(rp_info.rcomp_other_sources)
            )

            last_fw: fx.Node = rp_info.last_forward_access
            last_fw_info: NodeInfo = self.node_info[last_fw]
            last_fw_info.to_delete.append(rp)
            last_bw: fx.Node = rp_info.first_back_access
            last_bw_info: NodeInfo = self.node_info[last_bw]
            last_bw_info.to_recompute.append(rp)
        self.req_recomps = self.recomps

    def _initMSPS(self, candidates: List[fx.Node], placeholders: List[fx.Node]):
        def populate_sources_from_candidates(
            node: fx.Node, candidates: List[fx.Node], placeholders: List[fx.Node]
        ) -> Tuple[List[fx.Node], List[fx.Node], List[fx.Node]]:
            inp_nodes: List[fx.Node] = node.all_input_nodes
            activation_sources: List[fx.Node] = []
            param_sources: List[fx.Node] = []
            other_sources: List[fx.Node] = []
            for i_node in inp_nodes:
                if i_node in candidates:
                    activation_sources.append(i_node)
                elif i_node in placeholders:
                    param_sources.append(i_node)
                else:
                    other_sources.append(i_node)
                    a_srcs, p_srcs, o_srcs = populate_sources_from_candidates(
                        i_node, candidates, placeholders
                    )
                    activation_sources.extend(a_srcs)
                    param_sources.extend(p_srcs)
                    other_sources.extend(o_srcs)
            return (activation_sources, param_sources, other_sources)

        candidate_summaries: List[List[Any]] = []
        for cand in candidates:
            n_info: IntNodeInfo = self.node_info[cand]
            n_info.rcomp_outs = [cand]
            (
                n_info.rcomp_activation_sources,
                n_info.rcomp_param_sources,
                n_info.rcomp_other_sources,
            ) = populate_sources_from_candidates(cand, candidates, placeholders)
            r_time: float = 0
            for n in n_info.rcomp_other_sources:
                r_time += self.node_info[n].run_time

            n_info.exe_time = n_info.rcomp_time = r_time + n_info.run_time
            n_info.MSPS = n_info.memory_size / n_info.exe_time
            candidate_summaries.append([str(cand), n_info.exe_time, n_info.memory_size])
        headers: List[str] = ["Candidate", "Cand Exe Time(ms)", "Cand Mem Size(B)"]
        # print(tabulate.tabulate(candidate_summaries, headers=headers))

    def _get_max_MSPS_candidate(self, candidates: List[fx.Node]) -> fx.Node:
        max_cand: fx.Node = None
        max_MSPS: float = 0
        for cand in candidates:
            cand_info: IntNodeInfo = self.node_info[cand]
            if cand_info.MSPS > max_MSPS:
                max_MSPS = cand_info.MSPS
                max_cand = cand
        return max_cand

    def _recompute_overhead(self, t: fx.Node) -> float:
        return self.node_info[t].exe_time

    def activation_checkpointing(self):
        candidates: List[fx.Node] = list(self.intermediate_nodes)
        placeholders: List[fx.Node] = self._get_placeholders()
        mem_saving = self.max_peak_mem - MEM_LIMIT
        print("Required mem_savings: ", mem_saving)
        self._(candidates, placeholders)
        while mem_saving > 0:
            t = self._(candidates)
            print("Candidate: ", str(t), " selected for recompute")
            t_info: IntNodeInfo = self.node_info[t]
            exe_count = self._update_existing_recomps(t)
            self.recomps.add(t)
            candidates.remove(t)
            mem_saving -= t_info.memory_size
            self._update_rem_candidates(t, exe_count, candidates)
        self._prep_recomps()
        self._optimize_recomps()
        self._rewrite_graph()