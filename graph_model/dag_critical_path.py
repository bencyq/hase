# -*- coding: utf-8 -*-
"""
Task 6.1: DAG Critical Path Analyzer

提供两类输入构图：
1) 从 Phase 2 的 kernel JSON 构建 DAG（当前版本按内核顺序构建顺序依赖）
2) 从原始 ONNX 模型构建 DAG（基于张量生产/消费关系）

并提供 calculate_latency() 用于计算图总时延。
"""
import argparse
import json
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.logger import get_logger
from ort_analysis.ort_graph_parser import get_optimized_model

logger = get_logger("dag_critical_path")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DAG_DIR = os.path.join(SCRIPT_DIR, "model_DAG")


def _dedup_edges(edges: List[Tuple[str, str]]) -> List[List[str]]:
    seen = set()
    out = []
    for src, dst in edges:
        if src == dst:
            continue
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        out.append([src, dst])
    return out


def _topological_sort(nodes: List[str], edges: List[List[str]]) -> List[str]:
    indegree = {n: 0 for n in nodes}
    graph = defaultdict(list)

    for src, dst in edges:
        graph[src].append(dst)
        indegree[dst] = indegree.get(dst, 0) + 1
        indegree.setdefault(src, indegree.get(src, 0))

    queue = deque([n for n in nodes if indegree.get(n, 0) == 0])
    order = []
    while queue:
        cur = queue.popleft()
        order.append(cur)
        for nxt in graph.get(cur, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(nodes):
        raise ValueError("Graph contains cycle or disconnected invalid nodes.")
    return order


def build_dag_from_kernel_json(kernel_json_path: str) -> Dict:
    """
    从 Phase 2 kernel JSON 构建 DAG。
    说明：当前 JSON 不包含显式张量依赖，先按 kernel 顺序构建顺序 DAG。
    """
    with open(kernel_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    kernels = data.get("kernels", [])
    nodes = [k["node_name"] for k in kernels]
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    dag_edges = _dedup_edges(edges)
    topo_order = _topological_sort(nodes, dag_edges) if nodes else []

    node_meta = {
        k["node_name"]: {
            "kernel_type": k.get("kernel_type"),
            "fusion_desc": k.get("fusion_desc"),
        }
        for k in kernels
    }

    return {
        "source_type": "kernel_json",
        "source": os.path.abspath(kernel_json_path),
        "model_name": data.get("model_name", "unknown"),
        "nodes": nodes,
        "edges": dag_edges,
        "topological_order": topo_order,
        "node_meta": node_meta,
    }


def build_dag_from_model(onnx_path: str) -> Dict:
    """从原始 ONNX 模型构建 DAG（基于张量生产/消费依赖）。"""
    model = get_optimized_model(onnx_path)
    graph = model.graph
    nodes_proto = list(graph.node)

    nodes = []
    node_meta = {}
    for idx, node in enumerate(nodes_proto):
        node_name = node.name if node.name else "node_{}".format(idx)
        nodes.append(node_name)
        node_meta[node_name] = {"op_type": node.op_type}

    producer_by_tensor = {}
    for idx, node in enumerate(nodes_proto):
        node_name = node.name if node.name else "node_{}".format(idx)
        for out in node.output:
            producer_by_tensor[out] = node_name

    edges = []
    for idx, node in enumerate(nodes_proto):
        cur_name = node.name if node.name else "node_{}".format(idx)
        for inp in node.input:
            src = producer_by_tensor.get(inp)
            if src:
                edges.append((src, cur_name))

    dag_edges = _dedup_edges(edges)
    topo_order = _topological_sort(nodes, dag_edges) if nodes else []
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]

    return {
        "source_type": "onnx_model",
        "source": os.path.abspath(onnx_path),
        "model_name": model_name,
        "nodes": nodes,
        "edges": dag_edges,
        "topological_order": topo_order,
        "node_meta": node_meta,
    }


def save_dag(dag: Dict, output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dag, f, ensure_ascii=False, indent=2)
    return output_path


def calculate_latency(
    dag: Dict,
    kernel_latency_ms: Dict[str, float],
    sequential_stream0: bool = True,
) -> float:
    """
    计算图总时延：
    - sequential_stream0=True: 按拓扑顺序直接求和（初始版本）
    - False: 计算 DAG 最长路径时延
    """
    nodes = dag.get("nodes", [])
    edges = dag.get("edges", [])
    topo = dag.get("topological_order") or _topological_sort(nodes, edges)

    def node_latency(node_name: str) -> float:
        return float(kernel_latency_ms.get(node_name, 0.0))

    if sequential_stream0:
        return float(sum(node_latency(n) for n in topo))

    preds = defaultdict(list)
    for src, dst in edges:
        preds[dst].append(src)

    dist = {}
    for node in topo:
        best_prev = 0.0
        if preds.get(node):
            best_prev = max(dist[p] for p in preds[node])
        dist[node] = best_prev + node_latency(node)
    return max(dist.values()) if dist else 0.0


def _load_latency_map(latency_json_path: str) -> Dict[str, float]:
    with open(latency_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # 兼容 {"nodeA": 1.2, ...}
        if all(isinstance(v, (int, float)) for v in data.values()):
            return {str(k): float(v) for k, v in data.items()}
        # 兼容 {"latency_ms": {...}}
        if "latency_ms" in data and isinstance(data["latency_ms"], dict):
            return {str(k): float(v) for k, v in data["latency_ms"].items()}

    raise ValueError("Unsupported latency json format: {}".format(latency_json_path))


def _default_output_path(model_name: str) -> str:
    os.makedirs(DAG_DIR, exist_ok=True)
    return os.path.join(DAG_DIR, "{}_dag.json".format(model_name))


def main():
    parser = argparse.ArgumentParser(description="Task 6.1 DAG Critical Path Analyzer")
    parser.add_argument("--kernel-json", type=str, default=None, help="Phase2 kernel json 路径")
    parser.add_argument("--model", type=str, default=None, help="原始 ONNX 模型路径")
    parser.add_argument("--output", type=str, default=None, help="DAG 输出路径")
    parser.add_argument("--latency-json", type=str, default=None, help="节点时延映射 JSON")
    parser.add_argument(
        "--non-sequential",
        action="store_true",
        help="使用 DAG 最长路径计算（默认按顺序求和）",
    )
    args = parser.parse_args()

    if not args.kernel_json and not args.model:
        raise ValueError("请提供 --kernel-json 或 --model 其中之一。")

    if args.kernel_json:
        dag = build_dag_from_kernel_json(args.kernel_json)
    else:
        dag = build_dag_from_model(args.model)

    output_path = args.output or _default_output_path(dag["model_name"])
    save_dag(dag, output_path)
    logger.info("DAG 已生成: %s", output_path)
    logger.info("节点数=%d, 边数=%d", len(dag["nodes"]), len(dag["edges"]))

    if args.latency_json:
        latency_map = _load_latency_map(args.latency_json)
    else:
        # 未提供时延时，用 1.0ms 做快速连通性测试
        latency_map = {name: 1.0 for name in dag["nodes"]}

    total_ms = calculate_latency(
        dag,
        latency_map,
        sequential_stream0=not args.non_sequential,
    )
    logger.info("calculate_latency 结果: %.6f ms", total_ms)


if __name__ == "__main__":
    main()
