# Vector Search Benchmarks

Benchmark system for comparing CAGRA (GPU) vs Lucene HNSW (CPU) vector search algorithms.

## Setup

1. **Prerequisites:**
   - JDK 22+
   - CUDA libraries
   - Python 3.7+ with packages for post-run analysis and plots (required if you use `--run-benchmarks`, which triggers Pareto export/plotting):

   ```bash
   pip install pandas matplotlib numpy click pyyaml
   ```

   | Package | Used for |
   |---------|----------|
   | `pandas` | `data_export.py` — build/search CSVs and Pareto frontiers |
   | `matplotlib`, `numpy`, `click` | `plot_pareto.py` — throughput/latency plots |
   | `pyyaml` | YAML utilities in supporting scripts |

   See also [BenchmarkWithCAGRASearchPrototype.md](BenchmarkWithCAGRASearchPrototype.md) for the full CAGRA prototype setup.

2. **Set library paths:**
   ```bash
   export LD_LIBRARY_PATH="/path/to/cuvs/build:/path/to/cuda/lib64:/path/to/conda/lib:$LD_LIBRARY_PATH"
   ```

## Run benchmark


### cuVS-Lucene benchmarks

    ./run_sweep.sh --data-dir /data2/vsbench-datasets --datasets datasets.json --sweeps sweeps.json --configs-dir configs --results-dir results --run-benchmarks


### Solr benchmarks

    ./run_sweep.sh --data-dir /data2/vsbench-datasets --datasets datasets.json --mode solr --sweeps solr-sweeps.json --configs-dir configs --results-dir results --run-benchmarks


It builds Apache Solr's main branch and runs the benchmarks.


## Adding Datasets

Edit `datasets.json`:

## Creating Sweeps

Edit (or copy+edit) `sweep.json`:

## Visualization

./run_pareto_analysis.sh <benchmark-id or sweep-id>  <dataset>(already called in run_sweep.sh)
example: ./run_pareto_analysis.sh 3cNWY5 wiki10m

Serve the webui on port 8000:

    cd web-ui-new; python3 -m http.server
