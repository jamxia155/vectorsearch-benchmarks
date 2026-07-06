#!/bin/bash

# NVIDIA Pareto Analysis Workflow
# Converts benchmark results to NVIDIA format, runs Pareto analysis, and generates plots

set -e

cd "$(dirname "$0")" || exit 1

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <sweep_id> <dataset_name>"
    echo "Example: $0 3cNWY5 wiki10m"
    exit 1
fi

SWEEP_ID="$1"
DATASET_NAME="$2"

INPUT_DIR="results/${SWEEP_ID}"
OUTPUT_DIR="results/${SWEEP_ID}/${DATASET_NAME}"
INTERMEDIATE_DIR="results/${SWEEP_ID}/intermediate-files"
RESULTS_DIR="results"

echo "Processing sweep: ${SWEEP_ID}, dataset: ${DATASET_NAME}"

rm -rf "${INTERMEDIATE_DIR}" "${OUTPUT_DIR}/plots"

FIRST_RESULTS=$(find "${INPUT_DIR}/${DATASET_NAME}" -name "results.json" | head -1)
K=$(python3 -c "import json; print(json.load(open('${FIRST_RESULTS}'))['configuration']['topK'])")
N_QUERIES=$(python3 -c "import json; print(json.load(open('${FIRST_RESULTS}'))['configuration']['numQueriesToRun'])")

echo "Exporting full-parameter CSVs from results.json..."
python3 export_results_csv.py \
    --sweep-dir "${INPUT_DIR}/${DATASET_NAME}" \
    --output-dir "${INTERMEDIATE_DIR}" \
    --dataset "${DATASET_NAME}"

echo "Generating is_pareto files for Pareto optimal runs..."
python3 -c "
import os
import sys
import csv
import json
import glob

sys.path.append('.')
from export_results_csv import create_index_name as create_index_name_from_config

intermediate_dir = '${INTERMEDIATE_DIR}/${DATASET_NAME}'
results_dir = '${RESULTS_DIR}/${SWEEP_ID}/${DATASET_NAME}'

csv_patterns = [
    f'{intermediate_dir}/result/search/*throughput.csv',
    f'{intermediate_dir}/result/search/*latency.csv'
]

pareto_runs_by_algo = {}

for pattern in csv_patterns:
    csv_files = glob.glob(pattern)
    for csv_file in csv_files:
        algorithm = os.path.basename(csv_file).split(',')[0]

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            pareto_runs = list(reader)

        if algorithm not in pareto_runs_by_algo:
            pareto_runs_by_algo[algorithm] = {}

        for pareto_run in pareto_runs:
            index_name = pareto_run['index_name']
            if index_name not in pareto_runs_by_algo[algorithm]:
                pareto_runs_by_algo[algorithm][index_name] = pareto_run

print(f'Found Pareto optimal runs from CSV files:')
for algo, runs in pareto_runs_by_algo.items():
    print(f'  {algo}: {len(runs)} unique configurations')

for algorithm, pareto_indices in pareto_runs_by_algo.items():
    print(f'\\nProcessing {algorithm}...')

    benchmark_dirs = []
    for variant in [algorithm, algorithm.upper(), algorithm.lower()]:
        benchmark_dirs.extend(glob.glob(f'{results_dir}/{variant}-*'))

    if algorithm == 'CAGRA_HNSW':
        benchmark_dirs.extend(glob.glob(f'{results_dir}/cagra_hnsw-*'))
    elif algorithm == 'LUCENE_HNSW':
        benchmark_dirs.extend(glob.glob(f'{results_dir}/hnsw-*'))

    benchmark_dirs = list(set(benchmark_dirs))
    print(f'Found {len(benchmark_dirs)} result directories')

    index_to_dir = {}
    for benchmark_dir in benchmark_dirs:
        results_json_path = os.path.join(benchmark_dir, 'results.json')
        if os.path.exists(results_json_path):
            try:
                with open(results_json_path, 'r') as f:
                    results_data = json.load(f)

                config = results_data['configuration']
                algo_to_run = config.get('algoToRun')

                algorithm_match = algo_to_run == algorithm

                if algorithm_match:
                    index_name = create_index_name_from_config(config)
                    if index_name not in index_to_dir:
                        index_to_dir[index_name] = benchmark_dir
            except Exception as e:
                print(f'  Error processing {benchmark_dir}: {e}')

    print(f'Mapped {len(index_to_dir)} configurations')

    matched = 0
    unmatched = 0
    for index_name, pareto_run in pareto_indices.items():
        if index_name in index_to_dir:
            benchmark_dir = index_to_dir[index_name]
            is_pareto_file = os.path.join(benchmark_dir, 'is_pareto')

            with open(is_pareto_file, 'w') as f:
                f.write(f'Pareto optimal run\\n')
                f.write(f'Algorithm: {algorithm}\\n')
                f.write(f'Index: {index_name}\\n')
                f.write(f'Recall: {pareto_run[\"recall\"]}\\n')
                f.write(f'Throughput: {pareto_run[\"throughput\"]}\\n')
                f.write(f'Latency: {pareto_run[\"latency\"]}\\n')

            matched += 1
        else:
            unmatched += 1

    print(f'Matched {matched}/{len(pareto_indices)} runs')

print('\\nPareto file generation complete')
"

echo "Parameters: k=${K}, n_queries=${N_QUERIES}"

mkdir -p "${OUTPUT_DIR}/plots"

echo "Generating plots..."
python3 plot_pareto.py --dataset "${DATASET_NAME}" --dataset-path "${INTERMEDIATE_DIR}" --mode throughput --count "${K}" --n-queries "${N_QUERIES}" --output-filepath "${OUTPUT_DIR}/plots" --search
mv "${OUTPUT_DIR}/plots/search-${DATASET_NAME}-k${K}-n_queries${N_QUERIES}.png" "${OUTPUT_DIR}/plots/throughput-${DATASET_NAME}-k${K}-n_queries${N_QUERIES}.png"

python3 plot_pareto.py --dataset "${DATASET_NAME}" --dataset-path "${INTERMEDIATE_DIR}" --mode latency --count "${K}" --n-queries "${N_QUERIES}" --output-filepath "${OUTPUT_DIR}/plots" --search
mv "${OUTPUT_DIR}/plots/search-${DATASET_NAME}-k${K}-n_queries${N_QUERIES}.png" "${OUTPUT_DIR}/plots/latency-${DATASET_NAME}-k${K}-n_queries${N_QUERIES}.png"

python3 plot_pareto.py --dataset "${DATASET_NAME}" --dataset-path "${INTERMEDIATE_DIR}" --mode throughput --count "${K}" --n-queries "${N_QUERIES}" --output-filepath "${OUTPUT_DIR}/plots" --build

echo "Complete! Output saved to: ${OUTPUT_DIR}"
echo "Plots: ${OUTPUT_DIR}/plots/"
ls -la "${OUTPUT_DIR}/plots"/*.png

echo ""
echo "Exporting CSVs to csv-export..."
CSV_EXPORT_DIR="${INPUT_DIR}/csv-export/${DATASET_NAME}"
rm -rf "${CSV_EXPORT_DIR}"
mkdir -p "${CSV_EXPORT_DIR}"
find "${INTERMEDIATE_DIR}/${DATASET_NAME}" -maxdepth 1 -name '*.csv' -exec cp -t "${CSV_EXPORT_DIR}" {} +
echo "CSVs saved to: ${CSV_EXPORT_DIR}/"
ls -la "${CSV_EXPORT_DIR}/"

echo "Cleaning up intermediate files..."
rm -rf "${INTERMEDIATE_DIR}"
echo "Intermediate files cleaned up!"
echo ""
echo "Final output:"
echo "- Pareto optimal runs marked with is_pareto files"
echo "- Plots: ${OUTPUT_DIR}/plots/"
echo "- CSV export: ${CSV_EXPORT_DIR}/"