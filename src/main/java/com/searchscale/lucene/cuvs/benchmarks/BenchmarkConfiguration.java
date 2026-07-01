package com.searchscale.lucene.cuvs.benchmarks;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.CagraIndexParams.CodebookGen;
import com.nvidia.cuvs.CagraIndexParams.CudaDataType;
import com.nvidia.cuvs.CagraIndexParams.CuvsDistanceType;
import com.nvidia.cuvs.CagraSearchParams;
import com.searchscale.lucene.cuvs.benchmarks.LuceneCuvsBenchmarks.Codex;

public class BenchmarkConfiguration {

  public String benchmarkID;
  public String datasetFile;
  public int indexOfVector;
  public String vectorColName;
  public int numDocs;
  public int vectorDimension;
  public String queryFile;
  public int numQueriesToRun;
  public int numWarmUpQueries;
  public int flushFreq;
  public int topK;
  public int numIndexThreads;
  public int cuvsWriterThreads;
  public int queryThreads;
  public boolean createIndexInMemory;
  public boolean cleanIndexDirectory;
  public boolean saveResultsOnDisk;
  public String resultsDirectory;
  public boolean hasColNames;
  public Codex algoToRun;
  public String groundTruthFile;
  public String indexDirPath;
  public boolean loadVectorsInMemory;
  public boolean skipIndexing;
  public int forceMerge;
  public boolean enableTieredMerge;
  public boolean enableIndexWriterInfoStream;
  public double filterRejectRate =
      0.0; // 0.0 = no filter; fraction of docs rejected (higher = harder filter, steps of 0.001)

  // Lucene HNSW parameters
  public int hnswMaxConn = 16; // 16 default (max 512)
  public int hnswBeamWidth = 100; // 100 default (max 3200)
  public int hnswMergeThreads;

  // CAGRA parameters
  public int cagraIntermediateGraphDegree; // 128 default
  public int cagraGraphDegree; // 64 default
  public int cagraITopK;
  public int cagraSearchWidth;
  public int cagraThreadBlockSize; // 0 = auto; valid: 64, 128, 256, 512, 1024
  public int cagraMaxIterations; // 0 = auto
  public long cuvsWorkspacePoolSize =
      0; // initial workspace pool reservation per query thread in bytes (0 = disabled); pool grows
  // beyond this if needed
  public CagraSearchParams.SearchAlgo cagraSearchAlgo = CagraSearchParams.SearchAlgo.AUTO;
  public int cagraHnswLayers; // layers in CAGRA->HNSW conversion
  public int efSearch;
  public CagraGraphBuildAlgo cagraGraphBuildAlgo;

  // CAGRA IVF_PQ parameters
  public int cuVSIvfPqParamsRefinementRate = 1;
  public boolean cuVSIvfPqIndexParamsAddDataOnBuild = true;
  public CodebookGen cuVSIvfPqIndexParamsCodebookKind = CodebookGen.PER_SUBSPACE;
  public boolean cuVSIvfPqIndexParamsConservativeMemoryAllocation = false;
  public boolean cuVSIvfPqIndexParamsForceRandomRotation = false;
  public int cuVSIvfPqIndexParamsKmeansNIters = 20;
  public double cuVSIvfPqIndexParamsKmeansTrainsetFraction = 0.5;
  public int cuVSIvfPqIndexParamsMaxTrainPointsPerPqCode = 256;
  public CuvsDistanceType cuVSIvfPqIndexParamsMetric = CuvsDistanceType.L2Expanded;
  public float cuVSIvfPqIndexParamsMetricArg = 2.0f;
  public int cuVSIvfPqIndexParamsNLists = 1024;
  public int cuVSIvfPqIndexParamsPqBits = 8;
  public int cuVSIvfPqIndexParamsPqDim = 0;
  public CudaDataType cuVSIvfPqSearchParamsInternalDistanceDtype = CudaDataType.CUDA_R_32F;
  public CudaDataType cuVSIvfPqSearchParamsLutDtype = CudaDataType.CUDA_R_32F;
  public int cuVSIvfPqSearchParamsNProbes = 20;
  public double cuVSIvfPqSearchParamsPreferredShmemCarveout = 1.0;

  public boolean isLucene() {
    return Codex.LUCENE_HNSW.equals(algoToRun);
  }

  public boolean isCagra() {
    return Codex.CAGRA_HNSW.equals(algoToRun);
  }

  public boolean isCagraSearch() {
    return Codex.CAGRA_SEARCH.equals(algoToRun);
  }

  public boolean isCagraHNSWBinary() {
    return Codex.CAGRA_HNSW_BINARY.equals(algoToRun);
  }

  public boolean isCagraHNSWScalar() {
    return Codex.CAGRA_HNSW_SCALAR.equals(algoToRun);
  }

  public int getEffectiveEfSearch() {
    if (efSearch > 0) {
      return efSearch;
    }
    return Math.max(topK, (int) Math.ceil(topK * 1.5));
  }

  public String prettyString() {
    StringBuilder sb = new StringBuilder();
    sb.append("Benchmark ID: ").append(benchmarkID).append('\n');
    sb.append("Dataset file used is: ").append(datasetFile).append('\n');
    sb.append("Index of vector field is: ").append(indexOfVector).append('\n');
    sb.append("Name of the vector field is: ").append(vectorColName).append('\n');
    sb.append("Number of documents to be indexed are: ").append(numDocs).append('\n');
    sb.append("Number of dimensions are: ").append(vectorDimension).append('\n');
    sb.append("Query file used is: ").append(queryFile).append('\n');
    sb.append("Number of queries to run: ").append(numQueriesToRun).append('\n');
    sb.append("Number of warmup queries: ").append(numWarmUpQueries).append('\n');
    sb.append("Flush frequency (every n documents): ").append(flushFreq).append('\n');
    sb.append("TopK value is: ").append(topK).append('\n');
    sb.append("numIndexThreads is: ").append(numIndexThreads).append('\n');
    sb.append("Query threads: ").append(queryThreads).append('\n');
    sb.append("Create index in memory: ").append(createIndexInMemory).append('\n');
    sb.append("Clean index directory: ").append(cleanIndexDirectory).append('\n');
    sb.append("Save results on disk: ").append(saveResultsOnDisk).append('\n');
    sb.append("Has column names in the dataset file: ").append(hasColNames).append('\n');
    sb.append("algoToRun: ").append(algoToRun).append('\n');
    sb.append("Ground Truth file used is: ").append(groundTruthFile).append('\n');
    sb.append("index directory path is: ").append(indexDirPath).append('\n');
    sb.append("Load vectors in memory before indexing: ").append(loadVectorsInMemory).append('\n');
    sb.append("Skip indexing (and use existing index for search): ")
        .append(skipIndexing)
        .append('\n');
    sb.append("Do force merge while indexing documents [a value < 1 implies no force merge]: ")
        .append(forceMerge)
        .append('\n');
    sb.append("Enable TieredMerge: ").append(enableTieredMerge).append('\n');
    sb.append("Num HNSW merge threads: ").append(hnswMergeThreads).append('\n');
    sb.append("enableIndexWriterInfoStream: ").append(enableIndexWriterInfoStream).append('\n');
    sb.append("filterRejectRate: ").append(filterRejectRate).append('\n');

    sb.append("------- algo parameters ------\n");
    if (isLucene()) {
      sb.append("hnswMaxConn: ").append(hnswMaxConn).append('\n');
      sb.append("hnswBeamWidth: ").append(hnswBeamWidth).append('\n');
    } else {
      sb.append("cagraIntermediateGraphDegree: ").append(cagraIntermediateGraphDegree).append('\n');
      sb.append("cagraGraphDegree: ").append(cagraGraphDegree).append('\n');
      sb.append("cuvsWriterThreads: ").append(cuvsWriterThreads).append('\n');
      sb.append("cagraITopK: ").append(cagraITopK).append('\n');
      sb.append("cagraSearchWidth: ").append(cagraSearchWidth).append('\n');
      sb.append("cagraThreadBlockSize: ").append(cagraThreadBlockSize).append('\n');
      sb.append("cagraMaxIterations: ").append(cagraMaxIterations).append('\n');
      sb.append("cuvsWorkspacePoolSize: ").append(cuvsWorkspacePoolSize).append('\n');
      sb.append("cagraSearchAlgo: ").append(cagraSearchAlgo).append('\n');
      sb.append("cagraHnswLayers: ").append(cagraHnswLayers).append('\n');
      sb.append("cagraGraphBuildAlgo: ").append(cagraGraphBuildAlgo).append('\n');
    }
    return sb.toString();
  }

  @Override
  public String toString() {
    return prettyString();
  }

  public void debugPrintArguments() {
    // keep a single source of truth for printing
    System.out.print(prettyString());
  }
}
