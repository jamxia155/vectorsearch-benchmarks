# CuVS Lucene benchmarks with CAGRA search prototype

## Prerequisites

### Code
Get `cuVS`, `cuVS-Lucene`, and `vectorsearch-benchmarks` codebases from jamxia155's forks
```sh
git clone --branch jamxia-cagra-search-prototype https://github.com/jamxia155/cuvs.git
git clone --branch jamxia-cagra-search-prototype https://github.com/jamxia155/cuvs-lucene.git
git clone --branch jamxia-cagra-search-prototype https://github.com/jamxia155/vectorsearch-benchmarks.git
```

### Supporting tools
I am mentioning the versions below that I use on `ubuntu-24.04`

- cmake (4.3.0)
- apache maven (3.9.13)
- java (22)
- cuda (12.9)
- axel (faster alternative to wget for downloading stuff)
- ninja and nccl (used while building cuVS)
- nvtop

### Python (Pareto CSV export and plots)

After a sweep, `run_sweep.sh` calls `run_pareto_analysis.sh`, which needs:

- Python 3.7+
- **pandas** — `data_export.py` (build/search CSVs, Pareto frontiers)
- **matplotlib**, **numpy**, **click** — `plot_pareto.py` (throughput/latency plots)
- **pyyaml** — optional helpers elsewhere in the repo

Install once:

```sh
pip install pandas matplotlib numpy click pyyaml
```

Re-run analysis only (no re-benchmark) after a completed sweep:

```sh
cd vectorsearch-benchmarks
./run_pareto_analysis.sh <benchmark-id> <dataset-folder-name>
# Example: ./run_pareto_analysis.sh wOKdmU wiki1m
```

Plots land under `results/<benchmark-id>/<dataset>/plots/`. Full-parameter CSVs (raw + Pareto frontiers) are written to `results/<benchmark-id>/csv-export/<dataset>/` via `export_results_csv.py` (all sweep fields from each `results.json`).

You can get the above using the following:
```sh
sudo apt install -y axel ninja-build libnccl2 libnccl-dev nvtop
axel https://github.com/Kitware/CMake/releases/download/v4.3.0-rc1/cmake-4.3.0-rc1-linux-x86_64.tar.gz
axel https://download.java.net/java/GA/jdk22/830ec9fcccef480bb3e73fb7ecafe059/36/GPL/openjdk-22_linux-x64_bin.tar.gz
axel https://dlcdn.apache.org/maven/maven-3/3.9.13/binaries/apache-maven-3.9.13-bin.tar.gz
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-9
```

Extract maven, cmake, and java downloaded above and update the `PATH` and `LD_LIBRARY_PATH` environment variables:
```sh
export PATH="<CUDA_HOME, e.g. /usr/local/cuda/bin>:<MAVEN_HOME, e.g. /workspace/apache-maven-3.9.13/bin>:<CMAKE_HOME, e.g. /workspace/cmake-4.3.0-rc1-linux-x86_64/bin>:<JAVA22_HOME, e.g. /workspace/jdk-22/bin>:$PATH"
export LD_LIBRARY_PATH=<CUVS_REPO_DIR>/cpp/build/c:$LD_LIBRARY_PATH
```

Confirm the versions:
```sh
cmake --version && mvn --version && java --version && nvcc --version
```

Should see something like for the above:
```sh
cmake version 4.1.0

CMake suite maintained and supported by Kitware (kitware.com/cmake).
Apache Maven 3.9.11 (3e54c93a704957b63ee3494413a2b544fd3d825b)
Maven home: /home/narang/tools/apache-maven-3.9.11
Java version: 22, vendor: Oracle Corporation, runtime: /home/narang/tools/jdk-22
Default locale: en_US, platform encoding: UTF-8
OS name: "linux", version: "6.14.0-28-generic", arch: "amd64", family: "unix"
openjdk 22 2024-03-19
OpenJDK Runtime Environment (build 22+36-2370)
OpenJDK 64-Bit Server VM (build 22+36-2370, mixed mode, sharing)
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Tue_May_27_02:21:03_PDT_2025
Cuda compilation tools, release 12.9, V12.9.86
Build cuda_12.9.r12.9/compiler.36037853_0
```

Get the Wikipedia 1M/10Mx768 dataset files
```sh
mkdir data && cd data
axel https://data.rapids.ai/raft/datasets/wiki_all_1M/wiki_all_1M.tar
axel https://data.rapids.ai/raft/datasets/wiki_all_10M/wiki_all_10M.tar
```

## Build and run benchmarks

We need to build `cuVS` followed by `cuVS-Lucene` and then `vectorsearch-benchmarks`.

In the cuVS repo folder do the following to build `cuVS` and `cuVS-Java`
```sh
./build.sh libcuvs java
```
This will build `cuVS` and will install `cuVS-Java` maven artifacts in your local.

After the above succeeds, in the `cuVS-Lucene` repo folder do:
```sh
./build.sh
```
This will build `cuVS-Lucene` and will install maven artifacts in your local.

Now we can proceed to build `vectorsearch-benchmarks` and run sweeps.

Now the `vectorsearch-benchmarks` can run benchmarks using the following codecs. 
Notice the following keys in the `algorithms` in your configurations.

- `CAGRA_HNSW`: Uses the `Lucene101AcceleratedHNSWCodec` to index on GPU and search on CPU
- `LUCENE_HNSW`: Uses the Apache Lucene's `Lucene101Codec` to index on CPU and search on CPU
- `CAGRA_SEARCH`: Uses the `CuVS2510GPUSearchCodec` to index on GPU and search on GPU
- `CAGRA_HNSW_BINARY`: Uses the `LuceneAcceleratedHNSWBinaryQuantizedCodec`
- `CAGRA_HNSW_SCALAR`: Uses the `LuceneAcceleratedHNSWScalarQuantizedCodec`

Run sweeps as below (modify according to your local setup):

```sh
cd vectorsearch-benchmarks
CUDA_DEVICE_MAX_CONNECTIONS=<1 to 32, default 8; should size according to `queryThreads` in sweep config> ./run_sweep.sh --data-dir ../data --datasets datasets_test_1M.json --sweeps sweeps/test_1M.json --configs-dir configs --results-dir results --run-benchmarks
```

If needed, ensure files in data/wiki_all_1M are named as follows:
```sh
base.1M.fbin
groundtruth.1M.distances.fbin
groundtruth.1M.neighbors.ibin
queries.fbin
```
