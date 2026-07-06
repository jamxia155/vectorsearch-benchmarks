# Sweep definitions

Pass any file here to `run_sweep.sh` with `--sweeps sweeps/<file>.json`.

| File | Description |
|------|-------------|
| `default.json` | Upstream default sweep (wiki-10m, sift-1m) |
| `test_1M.json` | Small 1M wiki smoke (CAGRA_SEARCH + CAGRA_HNSW) |
| `cagra_search_grid_1M.json` | Full CAGRA_SEARCH NN_DESCENT grid (wiki 1M) |
| `cagra_search_grid.json` | CAGRA_SEARCH grid (10M dataset name) |
| `cagra_ivfpq_1M.json` | CAGRA_SEARCH + CAGRA_HNSW, IVF_PQ build (wiki 1M) |
| `cagra_ivfpq_1M_full.json` | Full CAGRA_HNSW IVF_PQ grid (wiki 1M) |
| `lucene_hnsw_1M_full.json` | Full LUCENE_HNSW grid (wiki 1M) |
| `ads10m_cagra_ivfpq_1536d.json` | CAGRA_HNSW IVF_PQ grid (ads 10M, 1536d, topK=200) |
| `ads10m_lucene_hnsw_1536d.json` | LUCENE_HNSW grid (ads 10M, 1536d, topK=200) |
| `ads10m_cagra_ivfpq_96d.json` | CAGRA_HNSW IVF_PQ grid (ads 10M, 96d, topK=1500) |
| `ads10m_lucene_hnsw_96d.json` | LUCENE_HNSW grid (ads 10M, 96d, topK=1500) |

Example:

```bash
./run_sweep.sh --data-dir /raid/workspace/data \
  --datasets datasets_test_1M.json \
  --sweeps sweeps/cagra_ivfpq_1M.json \
  --configs-dir configs_cagra_ivfpq_1m \
  --results-dir results \
  --run-benchmarks
```
