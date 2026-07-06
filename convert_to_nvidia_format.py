#!/usr/bin/env python3

import json
import os
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple


def create_index_name(config: Dict) -> str:
    """Create index name from configuration parameters"""
    algorithm = config.get('algoToRun', 'UNKNOWN')
    ef_search = config.get('efSearch', 0)

    if algorithm in ['LUCENE_HNSW', 'hnsw']:
        beam_width = config.get('hnswBeamWidth', 0)
        max_conn = config.get('hnswMaxConn', 0)
        return f"beam{beam_width}-conn{max_conn}-ef{ef_search}"
    elif algorithm in ['CAGRA_HNSW', 'cagra_hnsw']:
        graph_degree = config.get('cagraGraphDegree', 0)
        intermediate_degree = config.get('cagraIntermediateGraphDegree', 0)
        return f"ef{ef_search}-deg{graph_degree}-ideg{intermediate_degree}"
    elif algorithm in ['CAGRA_SEARCH', 'cagra_search']:
        graph_degree = config.get('cagraGraphDegree', 0)
        intermediate_degree = config.get('cagraIntermediateGraphDegree', 0)
        search_width = config.get('cagraSearchWidth', 0)
        query_threads = config.get('queryThreads', 0)
        return (
            f"ef{ef_search}-sw{search_width}-deg{graph_degree}-"
            f"ideg{intermediate_degree}-qt{query_threads}"
        )
    else:
        return f"ef{ef_search}"


def convert_results_to_nvidia_format(results_json_path: str, output_dir: str, dataset_name: str = None) -> Tuple[str, Optional[str]]:
    """Convert results.json to NVIDIA JSON format"""
    with open(results_json_path, 'r') as f:
        results_data = json.load(f)

    config = results_data['configuration']
    metrics = results_data['metrics']
    algorithm = config['algoToRun']

    if algorithm in ['cagra_hnsw', 'CAGRA_HNSW']:
        algorithm = 'CAGRA_HNSW'
    elif algorithm in ['hnsw', 'LUCENE_HNSW']:
        algorithm = 'LUCENE_HNSW'

    index_name = create_index_name(config)

    recall_key = next((key for key in metrics.keys() if 'recall-accuracy' in key.lower()), None)
    if not recall_key:
        raise KeyError("No recall-accuracy metric found")

    recall = float(metrics[recall_key]) / 100.0

    latency_key = next((key for key in metrics.keys() if 'mean-latency' in key.lower()), None)
    if not latency_key:
        raise KeyError("No mean-latency metric found")

    latency_ms = float(metrics[latency_key])
    throughput_key = next((key for key in metrics.keys() if 'query-throughput' in key.lower()), None)
    if throughput_key:
        throughput = float(metrics[throughput_key])
    else:
        throughput = 1000.0 / latency_ms if latency_ms > 0 else 0

    benchmark = {
        "name": f"{algorithm}/{index_name}",
        "real_time": latency_ms,
        "Recall": recall,
        "Latency": latency_ms,
        "items_per_second": throughput,
        "iterations": 1,
        "time_unit": "ms",
        "run_name": "run_1",
        "run_type": "iteration",
        "repetitions": 1,
        "repetition_index": 0,
        "family_index": 0,
        "per_family_instance_index": 0
    }

    if dataset_name is None:
        path_parts = Path(results_json_path).parts
        dataset_name = path_parts[-3] if len(path_parts) >= 3 else "unknown"

    k = config['topK']
    n_queries = config['numQueriesToRun']

    dataset_dir = Path(output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    search_filename = f"{algorithm},base,k{k},bs{n_queries},throughput.json"
    search_filepath = dataset_dir / search_filename

    if search_filepath.exists():
        with open(search_filepath, 'r') as f:
            data = json.load(f)
        data['benchmarks'].append(benchmark)
    else:
        data = {"benchmarks": [benchmark]}

    with open(search_filepath, 'w') as f:
        json.dump(data, f, indent=2)

    build_filepath = None
    build_time_key = next((key for key in metrics.keys() if 'indexing-time' in key.lower()), None)

    if build_time_key:
        build_time_ms = float(metrics[build_time_key])

        build_benchmark = {
            "name": f"{algorithm}/{index_name}",
            "real_time": build_time_ms,
            "iterations": 1,
            "time_unit": "ms",
            "run_name": "run_1",
            "run_type": "iteration",
            "repetitions": 1,
            "repetition_index": 0,
            "family_index": 0,
            "per_family_instance_index": 0
        }

        build_filename = f"{algorithm},base.json"
        build_filepath = dataset_dir / build_filename

        if build_filepath.exists():
            with open(build_filepath, 'r') as f:
                data = json.load(f)
            data['benchmarks'].append(build_benchmark)
        else:
            data = {"benchmarks": [build_benchmark]}

        with open(build_filepath, 'w') as f:
            json.dump(data, f, indent=2)

    return str(search_filepath), str(build_filepath) if build_filepath else None


def convert_sweep_to_nvidia_format(sweep_dir: str, output_dir: str, dataset_name: str = None) -> List[str]:
    """Convert entire sweep to NVIDIA format"""
    converted_files = []

    for root, dirs, files in os.walk(sweep_dir):
        if 'results.json' in files:
            results_path = os.path.join(root, 'results.json')

            try:
                search_file, build_file = convert_results_to_nvidia_format(
                    results_path, output_dir, dataset_name
                )
                converted_files.append(search_file)
                if build_file:
                    converted_files.append(build_file)
                print(f"Converted: {root}")
            except Exception as e:
                print(f"Error converting {root}: {e}")

    return converted_files


def main():
    parser = argparse.ArgumentParser(description='Convert benchmark results to NVIDIA JSON format')
    parser.add_argument('--sweep-dir', required=True, help='Directory containing sweep results')
    parser.add_argument('--output-dir', required=True, help='Output directory for NVIDIA format files')
    parser.add_argument('--dataset', help='Dataset name (auto-detected if not provided)')

    args = parser.parse_args()

    converted_files = convert_sweep_to_nvidia_format(args.sweep_dir, args.output_dir, args.dataset)
    print(f"Converted {len(converted_files)} files")

    if converted_files:
        print("Converted files:")
        for file_path in converted_files:
            print(f"  {file_path}")


if __name__ == "__main__":
    main()
