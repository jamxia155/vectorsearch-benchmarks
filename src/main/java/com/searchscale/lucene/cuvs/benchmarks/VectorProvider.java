package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;

/**
 * Interface for providing vectors from various sources (MapDB, direct file streaming, etc.)
 */
public interface VectorProvider {
  /**
   * Get the vector at the specified index
   */
  float[] get(int index) throws IOException;

  /**
   * Fill {@code dst} with the vector at the specified index, avoiding a per-call allocation. The
   * default copies from {@link #get(int)}; providers backed by a reusable buffer should override to
   * unpack directly. Callers may reuse {@code dst} once the vector has been consumed.
   */
  default void get(int index, float[] dst) throws IOException {
    float[] v = get(index);
    System.arraycopy(v, 0, dst, 0, dst.length);
  }

  /**
   * Get the total number of vectors
   */
  int size();

  /**
   * Close any resources
   */
  void close() throws IOException;
}
