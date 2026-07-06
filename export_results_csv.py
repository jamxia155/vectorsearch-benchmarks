#!/usr/bin/env python3
"""Export benchmark results.json files to analysis CSVs with full sweep parameters."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_export import get_frontier

# First five columns match plot_pareto.py comma-split indices (recall=2, throughput=3, latency=4).
CSV_COLUMNS = [
    "algo_name",
    "index_name",
    "recall",
    "throughput",
    "latency",
    "threads",
    "cpu_time",
    "end_to_end",
    "topk",
    "k",
    "n_queries",
    "persistent",
    "search_width",
    "total_queries",
    "build time",
    "build threads",
    "build cpu_time",
    "build GPU",
    "graph_degree",
    "intermediate_graph_degree",
    "graph_build_algo",
    "ivf_nlists",
    "ivf_pq_dim",
    "ivf_pq_bits",
    "ivf_kmeans_iters",
    "ivf_nprobes",
    "ivf_refinement_rate",
    "label",
    "run_directory",
]

# Maps CSV column names to keys in results.json configuration.
IVF_PQ_CONFIG_KEYS = {
    "ivf_nlists": "cuVSIvfPqIndexParamsNLists",
    "ivf_pq_dim": "cuVSIvfPqIndexParamsPqDim",
    "ivf_pq_bits": "cuVSIvfPqIndexParamsPqBits",
    "ivf_kmeans_iters": "cuVSIvfPqIndexParamsKmeansNIters",
    "ivf_nprobes": "cuVSIvfPqSearchParamsNProbes",
    "ivf_refinement_rate": "cuVSIvfPqParamsRefinementRate",
}


def _metric(metrics: Dict[str, Any], suffix: str) -> Optional[float]:
    suffix_lower = suffix.lower()
    for key, value in metrics.items():
        if suffix_lower in key.lower():
            if value is None:
                return None
            if isinstance(value, str):
                if value.lower() == "nan":
                    return float("nan")
                try:
                    return float(value)
                except ValueError:
                    return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def create_build_index_name(config: Dict[str, Any]) -> str:
    """Index identity for build metrics (excludes query-thread suffix)."""
    name = create_index_name(config)
    if config.get("algoToRun") in ["CAGRA_SEARCH", "cagra_search"]:
        return re.sub(r"-qt\d+$", "", name)
    return name


def create_index_name(config: Dict[str, Any]) -> str:
    algorithm = config.get("algoToRun", "UNKNOWN")
    ef_search = config.get("efSearch", 0)

    if algorithm in ["LUCENE_HNSW", "hnsw"]:
        beam_width = config.get("hnswBeamWidth", 0)
        max_conn = config.get("hnswMaxConn", 0)
        return f"beam{beam_width}-conn{max_conn}-ef{ef_search}"
    if algorithm in ["CAGRA_HNSW", "cagra_hnsw"]:
        graph_degree = config.get("cagraGraphDegree", 0)
        intermediate_degree = config.get("cagraIntermediateGraphDegree", 0)
        return f"ef{ef_search}-deg{graph_degree}-ideg{intermediate_degree}"
    if algorithm in ["CAGRA_SEARCH", "cagra_search"]:
        graph_degree = config.get("cagraGraphDegree", 0)
        intermediate_degree = config.get("cagraIntermediateGraphDegree", 0)
        search_width = config.get("cagraSearchWidth", 0)
        query_threads = config.get("queryThreads", 0)
        return (
            f"ef{ef_search}-sw{search_width}-deg{graph_degree}-"
            f"ideg{intermediate_degree}-qt{query_threads}"
        )
    return f"ef{ef_search}"


def _uses_ivf_pq_graph_build(config: Dict[str, Any]) -> bool:
    return config.get("cagraGraphBuildAlgo") == "IVF_PQ"


def ivf_pq_fields(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract IVF-PQ sweep fields; use NaN when graph build is not IVF_PQ."""
    graph_build_algo = config.get("cagraGraphBuildAlgo")
    if not _uses_ivf_pq_graph_build(config):
        return {
            "graph_build_algo": graph_build_algo,
            **{column: float("nan") for column in IVF_PQ_CONFIG_KEYS},
        }

    fields: Dict[str, Any] = {"graph_build_algo": graph_build_algo}
    for column, config_key in IVF_PQ_CONFIG_KEYS.items():
        value = config.get(config_key)
        fields[column] = float("nan") if value is None else value
    return fields


def create_label(config: Dict[str, Any]) -> str:
    algo = config.get("algoToRun", "")
    parts = [
        f"ef={config.get('efSearch', '')}",
        f"sw={config.get('cagraSearchWidth', '')}",
        f"gd={config.get('cagraGraphDegree', '')}",
        f"ig={config.get('cagraIntermediateGraphDegree', '')}",
        f"qt={config.get('queryThreads', '')}",
    ]
    if _uses_ivf_pq_graph_build(config):
        parts.extend(
            [
                f"lists={config.get('cuVSIvfPqIndexParamsNLists', '')}",
                f"probes={config.get('cuVSIvfPqSearchParamsNProbes', '')}",
                f"refine={config.get('cuVSIvfPqParamsRefinementRate', '')}",
            ]
        )
    return f"{algo} " + " ".join(parts)


def row_from_results(results_path: str) -> Dict[str, Any]:
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = data["configuration"]
    metrics = data.get("metrics", {})
    algo = config.get("algoToRun", "UNKNOWN")

    recall_pct = _metric(metrics, "recall-accuracy")
    recall = recall_pct / 100.0 if recall_pct is not None and not math.isnan(recall_pct) else float("nan")

    throughput = _metric(metrics, "query-throughput")
    latency_ms = _metric(metrics, "mean-latency")
    cpu_time = _metric(metrics, "mean-retrieval-latency")
    end_to_end = _metric(metrics, "query-time")
    build_time = _metric(metrics, "indexing-time")

    top_k = config.get("topK")
    total_queries = config.get("numQueriesToRun")
    warmup = config.get("numWarmUpQueries", 0)
    timed_queries = (total_queries - warmup) if total_queries is not None and warmup is not None else None

    persistent = not config.get("createIndexInMemory", False)

    return {
        "algo_name": algo,
        "index_name": create_index_name(config),
        "recall": recall,
        "throughput": throughput if throughput is not None else float("nan"),
        "latency": latency_ms if latency_ms is not None else float("nan"),
        "threads": config.get("queryThreads"),
        "cpu_time": cpu_time if cpu_time is not None else float("nan"),
        "end_to_end": end_to_end if end_to_end is not None else float("nan"),
        "topk": top_k,
        "k": top_k,
        "n_queries": timed_queries,
        "persistent": persistent,
        "search_width": config.get("cagraSearchWidth"),
        "total_queries": total_queries,
        "build time": build_time if build_time is not None else float("nan"),
        "build threads": config.get("numIndexThreads"),
        "build cpu_time": float("nan"),
        "build GPU": build_time if build_time is not None else float("nan"),
        "graph_degree": config.get("cagraGraphDegree"),
        "intermediate_graph_degree": config.get("cagraIntermediateGraphDegree"),
        **ivf_pq_fields(config),
        "label": create_label(config),
        "run_directory": config.get("resultsDirectory", results_path),
    }


def collect_rows(sweep_dir: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root, _, files in os.walk(sweep_dir):
        if "results.json" not in files:
            continue
        path = os.path.join(root, "results.json")
        try:
            rows.append(row_from_results(path))
        except Exception as exc:
            print(f"Error processing {path}: {exc}")
    return rows


def export_sweep_results(
    sweep_dir: str,
    output_dir: str,
    dataset_name: str,
) -> Dict[str, str]:
    rows = collect_rows(sweep_dir)
    if not rows:
        raise RuntimeError(f"No results.json files under {sweep_dir}")

    df = pd.DataFrame(rows)[CSV_COLUMNS]
    k = int(df["topk"].iloc[0])
    n_queries = int(df["total_queries"].iloc[0])

    dataset_out = Path(output_dir) / dataset_name
    dataset_out.mkdir(parents=True, exist_ok=True)
    plot_search_dir = dataset_out / "result" / "search"
    plot_build_dir = dataset_out / "result" / "build"
    plot_search_dir.mkdir(parents=True, exist_ok=True)
    plot_build_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, str] = {}

    for algo_name, algo_df in df.groupby("algo_name", sort=False):
        prefix = f"{algo_name},base,k{k},bs{n_queries}"

        raw_path = dataset_out / f"{prefix},raw.csv"
        algo_df.to_csv(raw_path, index=False)
        written["raw"] = str(raw_path)

        throughput_frontier = get_frontier(algo_df, "throughput")
        throughput_path = dataset_out / f"{prefix},throughput.csv"
        throughput_frontier.to_csv(throughput_path, index=False)
        written["throughput"] = str(throughput_path)

        latency_frontier = get_frontier(algo_df, "latency")
        latency_path = dataset_out / f"{prefix},latency.csv"
        latency_frontier.to_csv(latency_path, index=False)
        written["latency"] = str(latency_path)

        algo_df.to_csv(plot_search_dir / raw_path.name, index=False)
        throughput_frontier.to_csv(plot_search_dir / throughput_path.name, index=False)
        latency_frontier.to_csv(plot_search_dir / latency_path.name, index=False)

        build_df = algo_df[algo_df["build time"].notna()].copy()
        if not build_df.empty:
            build_export = build_df.copy()
            build_export["index_name"] = build_export.apply(
                lambda row: re.sub(r"-qt\d+$", "", row["index_name"]),
                axis=1,
            )
            build_export = build_export[
                ["algo_name", "index_name", "build time"]
            ].drop_duplicates(subset=["algo_name", "index_name"])
            build_path = dataset_out / f"{algo_name},base.csv"
            build_export.to_csv(build_path, index=False)
            build_plot_path = plot_build_dir / build_path.name
            build_export.to_csv(build_plot_path, index=False)
            written["build"] = str(build_path)

    print(f"Exported {len(df)} runs for dataset {dataset_name}")
    print(f"  Algorithms: {', '.join(df['algo_name'].unique())}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Export results.json to analysis CSVs")
    parser.add_argument("--sweep-dir", required=True, help="Directory containing per-run results")
    parser.add_argument("--output-dir", required=True, help="Output directory (e.g. intermediate-files)")
    parser.add_argument("--dataset", required=True, help="Dataset subdirectory name")
    args = parser.parse_args()

    export_sweep_results(args.sweep_dir, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()
