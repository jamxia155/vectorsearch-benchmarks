
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

# Parameters that only affect search, not index construction.
# All other variant parameters are treated as build-only, so the index is
# shared across every search-param combination that belongs to the same
# build-param combination.
#
# CuVS IVF-PQ params (index + search + refinement) are NOT listed here:
# for CAGRA graph build algorithms (IVF_PQ), they are consumed when the
# CAGRA graph is constructed, not at HNSW/CAGRA query time.
SEARCH_ONLY_PARAMS = {
    'efSearch',
    'cagraITopK',
    'cagraSearchWidth',
    'cagraThreadBlockSize',
    'cagraSearchAlgo',
    'topK',
    'queryThreads',
    'numQueriesToRun',
    'numWarmUpQueries',
    'filterRejectRate',
}

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
        if not isinstance(value, list):
            invariants[param] = value
        else:
            variants[param] = value
    for algo in sweeps[sweep].get("algorithms", []):
        algorithms = sweeps[sweep].get("algorithms", [])
        algo_variants = variants.copy()
        algo_invariants = invariants.copy()
        algo_invariants["algoToRun"] = algo

        for param, value in algorithms[algo].items():
            if param not in ["params"]:
                if not isinstance(value, list):
                    algo_invariants[param] = value
                else:
                    algo_variants[param] = value

        if algo_variants:
            # Split variants into build-time and search-time parameters.
            build_variant_keys = [k for k in algo_variants if k not in SEARCH_ONLY_PARAMS]
            build_variant_values = [algo_variants[k] for k in build_variant_keys]
            search_variant_keys = [k for k in algo_variants if k in SEARCH_ONLY_PARAMS]
            search_variant_values = [algo_variants[k] for k in search_variant_keys]

            build_combinations = list(itertools.product(*build_variant_values)) if build_variant_keys else [()]
            search_combinations = list(itertools.product(*search_variant_values)) if search_variant_keys else [()]

            for build_combo in build_combinations:
                build_variants = dict(zip(build_variant_keys, build_combo))

                # Skip invalid build-param combinations.
                if 'cagraIntermediateDegree' in build_variants and 'cagraGraphDegree' in build_variants:
                    if build_variants['cagraIntermediateDegree'] < build_variants['cagraGraphDegree']:
                        print(f"\t\tSkipping combination: cagraIntermediateDegree ({build_variants['cagraIntermediateDegree']}) < cagraGraphDegree ({build_variants['cagraGraphDegree']})")
                        continue
                if 'hnswMaxConn' in build_variants and 'hnswBeamWidth' in build_variants:
                    if build_variants['hnswMaxConn'] > build_variants['hnswBeamWidth']:
                        print(f"\t\tSkipping combination: hnswMaxConn ({build_variants['hnswMaxConn']}) > hnswBeamWidth ({build_variants['hnswBeamWidth']})")
                        continue

                # The index directory is identified by build params only, so all
                # search-param variants for this build share a single on-disk index.
                base_hash = hashlib.md5(json.dumps(build_variants, sort_keys=True).encode()).hexdigest()[:8]

                for search_idx, search_combo in enumerate(search_combinations):
                    search_variants = dict(zip(search_variant_keys, search_combo))

                    current_variants = {**build_variants, **search_variants}

                    # Append a zero-padded sequence index when there are search
                    # variants so that alphabetical filename order matches
                    # generation order (build config first, search-only configs
                    # after). A plain hash suffix would give arbitrary ordering.
                    if search_variant_keys:
                        hash_id = f"{base_hash}-s{search_idx:03d}"
                    else:
                        hash_id = base_hash

                    config = algo_invariants.copy()
                    config.update(current_variants)

                    # Only the first search-param combination builds the index.
                    if search_idx > 0:
                        config['skipIndexing'] = True

                    # Clean up the index directory after the last search-param
                    # combination for this build; keep it for all earlier ones.
                    config['cleanIndexDirectory'] = (search_idx == len(search_combinations) - 1)

                    # Point all configs in this build group at the same index directory.
                    index_dir = f"cuvsIndex-{base_hash}"
                    if 'hnswIndexDirPath' in config:
                        config['hnswIndexDirPath'] = f"hnswIndex-{base_hash}"
                    if 'cuvsIndexDirPath' in config:
                        config['cuvsIndexDirPath'] = index_dir
                    # LuceneCuvsBenchmarks reads indexDirPath (not cuvsIndexDirPath).
                    if 'indexDirPath' in config:
                        config['indexDirPath'] = index_dir

                    filename = f"{algo}-{hash_id}.json"
                    sweep_dir = f"{args.configs_dir}/{sweep}"
                    filepath = f"{sweep_dir}/{filename}"
                    os.makedirs(sweep_dir, exist_ok=True)
                    with open(filepath, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"\tGenerated config file: {filepath}")


    print("----------------------")
