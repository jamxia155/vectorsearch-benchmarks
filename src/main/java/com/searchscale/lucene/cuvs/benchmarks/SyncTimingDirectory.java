/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;
import java.util.Collection;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;

/**
 * Wraps a {@link Directory} to attribute the cost of durable syncs (the fsync work Lucene performs
 * inside {@code IndexWriter.commit()}) to dedicated {@link StageTimers} stages. On networked storage
 * this durable-sync tail can dominate commit while remaining invisible to the in-writer stage
 * timers, which only cover the buffered writes into the OS page cache.
 */
public final class SyncTimingDirectory extends FilterDirectory {

  public SyncTimingDirectory(Directory in) {
    super(in);
  }

  @Override
  public void sync(Collection<String> names) throws IOException {
    long bytes = 0L;
    for (String name : names) {
      try {
        bytes += fileLength(name);
      } catch (IOException e) {
        // fileLength is only used to report throughput; fall back to 0 if unavailable.
      }
    }
    long ts = StageTimers.start();
    in.sync(names);
    StageTimers.stop("durable-sync [DISK; fsync]", ts, bytes);
  }

  @Override
  public void syncMetaData() throws IOException {
    long ts = StageTimers.start();
    in.syncMetaData();
    StageTimers.stop("durable-sync-meta [DISK; fsync]", ts);
  }
}
