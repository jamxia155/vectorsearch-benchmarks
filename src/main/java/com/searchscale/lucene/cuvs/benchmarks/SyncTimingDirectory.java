/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.searchscale.lucene.cuvs.benchmarks;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;

/**
 * Wraps a {@link Directory} to attribute the cost of durable syncs (the fsync work Lucene performs
 * inside {@code IndexWriter.commit()}) to dedicated {@link StageTimers} stages. On networked storage
 * this durable-sync tail can dominate commit while remaining invisible to the in-writer stage
 * timers, which only cover the buffered writes into the OS page cache.
 */
public final class SyncTimingDirectory extends FilterDirectory {

  // Files already fsync'd over this wrapper's lifetime. Sharing one wrapper across the K sequential
  // IndexWriters of a partitioned build lets us skip re-syncing prior segments: each fresh writer's
  // commit would otherwise re-fsync every existing segment file (O(K^2) work). Re-fsyncing an
  // already-durable, immutable segment file is a no-op for durability, so skipping it is safe; the
  // commit point (segments_N) gets a fresh name each commit, so it is never skipped.
  private final Set<String> synced = ConcurrentHashMap.newKeySet();

  public SyncTimingDirectory(Directory in) {
    super(in);
  }

  @Override
  public void sync(Collection<String> names) throws IOException {
    List<String> toSync = new ArrayList<>(names.size());
    for (String name : names) {
      if (synced.add(name)) {
        toSync.add(name);
      }
    }
    if (toSync.isEmpty()) {
      return;
    }
    long bytes = 0L;
    for (String name : toSync) {
      try {
        bytes += fileLength(name);
      } catch (IOException e) {
        // fileLength is only used to report throughput; fall back to 0 if unavailable.
      }
    }
    long ts = StageTimers.start();
    in.sync(toSync);
    StageTimers.stop("durable-sync [DISK; fsync]", ts, bytes);
  }

  @Override
  public void syncMetaData() throws IOException {
    long ts = StageTimers.start();
    in.syncMetaData();
    StageTimers.stop("durable-sync-meta [DISK; fsync]", ts);
  }
}
