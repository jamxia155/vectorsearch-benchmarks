
import itertools
import argparse
import sys
import json
import hashlib
import os
import shutil
parser = argparse.ArgumentParser(description='Generate combinations script')
parser.add_argument('--data-dir', required=True, help='Data directory path')
parser.add_argument('--datasets', required=True, help='Datasets JSON file path')
parser.add_argument('--sweeps', required=True, help='Sweeps JSON file path')
parser.add_argument('--configs-dir', required=True, help='Configs output directory path')

args = parser.parse_args()

print("Arguments captured:")
print(f"data-dir: {args.data_dir}")
print(f"datasets: {args.datasets}")
print(f"sweeps: {args.sweeps}")
print(f"configs-dir: {args.configs_dir}")
print("----------------------")

sweeps = json.load(open(args.sweeps))
datasets = json.load(open(args.datasets))

# Clean configs directory
if os.path.exists(args.configs_dir):
    shutil.rmtree(args.configs_dir)

for sweep in sweeps:
    # Get dataset information for this sweep
    dataset_name = sweeps[sweep]["dataset"]
    dataset_info = datasets["datasets"][dataset_name]
    variants={}
    invariants={}

    # Add dataset-specific parameters to invariants
    invariants["datasetFile"] = f"{args.data_dir}/{dataset_name}/{dataset_info['base_file']}"
    invariants["queryFile"] = f"{args.data_dir}/{dataset_name}/{dataset_info['query_file']}"
    invariants["groundTruthFile"] = f"{args.data_dir}/{dataset_name}/{dataset_info['ground_truth_file']}"
    invariants["vectorDimension"] = dataset_info["vector_dimension"]
    print("sweep: " + sweep)
    for param, value in sweeps[sweep].get("common-params", {}).items():
        # efSearch is always passed through as a list — Java handles the iteration
        if param == 'efSearch':
            # Ensure it's always a list
            invariants[param] = value if isinstance(value, list) else [value]
        elif not isinstance(value, list):
            invariants[param] = value
        else:
            variants[param] = value
    # Normalize algorithms to a list of (name, params) tuples.
    # Supports both the original dict format:
    #   "algorithms": { "LUCENE_HNSW": {...}, "CAGRA_HNSW": {...} }
    # and the new list format (allows duplicate algorithm names):
    #   "algorithms": [ { "name": "LUCENE_HNSW", ... }, { "name": "LUCENE_HNSW", ... } ]
    raw_algorithms = sweeps[sweep].get("algorithms", [])
    if isinstance(raw_algorithms, dict):
        algo_list = [(name, params) for name, params in raw_algorithms.items()]
    else:
        algo_list = [(entry.pop("name"), entry) for entry in [dict(e) for e in raw_algorithms]]

    for algo, algo_params in algo_list:
        algo_variants = variants.copy()
        algo_invariants = invariants.copy()
        algo_invariants["algoToRun"] = algo

        for param, value in algo_params.items():
            if param not in ["params"]:
                # efSearch is always passed through as a list — Java handles the iteration
                if param == 'efSearch':
                    algo_invariants[param] = value if isinstance(value, list) else [value]
                elif not isinstance(value, list):
                    algo_invariants[param] = value
                else:
                    algo_variants[param] = value

        # Generate all combination of variants. For each combination, generate a hashed ID, and a file with the
        # name pattern as <sweep>-<algo>-<hash>.json. The file should contain the invariants as is, and the variants as the current combination.
        if algo_variants:
            variant_keys = list(algo_variants.keys())
            variant_values = list(algo_variants.values())
            for combination in itertools.product(*variant_values):
                current_variants = dict(zip(variant_keys, combination))
                
                config = algo_invariants.copy()
                config.update(current_variants)

                # Skip if cagraIntermediateGraphDegree < cagraGraphDegree
                # (CAGRA silently clamps graphDegree down to intermediateGraphDegree,
                # which produces duplicate test runs)
                if config.get('cagraIntermediateGraphDegree', float('inf')) < config.get('cagraGraphDegree', 0):
                    print(f"\t\tSkipping combination: cagraIntermediateGraphDegree ({config['cagraIntermediateGraphDegree']}) < cagraGraphDegree ({config['cagraGraphDegree']})")
                    continue
                
                # Skip if hnswMaxConn > hnswBeamWidth
                if config.get('hnswMaxConn', 0) > config.get('hnswBeamWidth', float('inf')):
                    print(f"\t\tSkipping combination: hnswMaxConn ({config['hnswMaxConn']}) > hnswBeamWidth ({config['hnswBeamWidth']})")
                    continue

                # Set indexDirPath based on hash
                hash_input = {k: v for k, v in config.items() if k != 'indexDirPath'}
                hash_id = hashlib.md5(json.dumps(hash_input, sort_keys=True, default=str).encode()).hexdigest()[:8]
                config['indexDirPath'] = os.path.join(config.get('indexDirPath', ''), f"index-{hash_id}")

                filename = f"{algo}-{hash_id}.json"
                sweep_dir = f"{args.configs_dir}/{sweep}"
                filepath = f"{sweep_dir}/{filename}"
                os.makedirs(sweep_dir, exist_ok=True)
                with open(filepath, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"\tGenerated config file: {filepath}")
        else:
            # No variants at all, just generate a single config
            config = algo_invariants.copy()

            # Set indexDirPath based on hash
            hash_input = {k: v for k, v in config.items() if k != 'indexDirPath'}
            hash_id = hashlib.md5(json.dumps(hash_input, sort_keys=True, default=str).encode()).hexdigest()[:8]
            config['indexDirPath'] = os.path.join(config.get('indexDirPath', ''), f"index-{hash_id}")

            filename = f"{algo}-{hash_id}.json"
            sweep_dir = f"{args.configs_dir}/{sweep}"
            filepath = f"{sweep_dir}/{filename}"
            os.makedirs(sweep_dir, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"\tGenerated config file: {filepath}")
        
        
    print("----------------------")
