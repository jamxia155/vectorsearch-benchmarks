package com.searchscale.lucene.cuvs.benchmarks;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

// TODO: The three static methods have a lot of common logic, ideally should be combined as just
// one.
public class FBIvecsReader {

  private static final Logger log = LoggerFactory.getLogger(FBIvecsReader.class.getName());

  public static int getDimension(InputStream fc) throws IOException {
    byte[] b = fc.readNBytes(4);
    ByteBuffer bb = ByteBuffer.wrap(b);
    bb.order(ByteOrder.LITTLE_ENDIAN);
    int dimension = bb.getInt();
    return dimension;
  }

  public static void readFvecs(String filePath, int numRows, List<float[]> vectors) {
    log.info("Reading {} from file: {}", numRows, filePath);

    try {
      InputStream is = null;

      if (filePath.endsWith("fvecs.gz")) {
        is = new GZIPInputStream(new FileInputStream(filePath));
      } else if (filePath.endsWith(".fvecs")) {
        is = new FileInputStream(filePath);
      }

      int dimension = getDimension(is);
      float[] row = new float[dimension];
      int count = 0;
      int rc = 0;

      while (is.available() != 0) {
        ByteBuffer bbf = ByteBuffer.wrap(is.readNBytes(4));
        bbf.order(ByteOrder.LITTLE_ENDIAN);
        row[rc++] = bbf.getFloat();

        if (rc == dimension) {
          vectors.add(row);
          count += 1;
          rc = 0;
          row = new float[dimension];

          // Skip last 4 bytes.
          is.readNBytes(4);

          if (count % 1000 == 0) {
            System.out.print(".");
          }

          if (numRows != -1 && count == numRows) {
            break;
          }
        }
      }
      System.out.println();
      is.close();
      log.info("Reading complete. Read {} vectors.", count);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static ArrayList<int[]> readIvecs(String filePath, int numRows) {
    log.info("Reading {} from file: {}", numRows, filePath);
    ArrayList<int[]> vectors = new ArrayList<int[]>();

    try {
      InputStream is = null;

      if (filePath.endsWith("ivecs.gz")) {
        is = new GZIPInputStream(new FileInputStream(filePath));
      } else if (filePath.endsWith(".ivecs")) {
        is = new FileInputStream(filePath);
      }

      int dimension = getDimension(is);
      int[] row = new int[dimension];
      int count = 0;
      int rc = 0;

      while (is.available() != 0) {

        ByteBuffer bbf = ByteBuffer.wrap(is.readNBytes(4));
        bbf.order(ByteOrder.LITTLE_ENDIAN);
        row[rc++] = bbf.getInt();

        if (rc == dimension) {
          vectors.add(row);
          count += 1;
          rc = 0;
          row = new int[dimension];

          // Skip last 4 bytes.
          is.readNBytes(4);

          if (count % 1000 == 0) {
            System.out.print(".");
          }

          if (numRows != -1 && count == numRows) {
            break;
          }
        }
      }
      System.out.println();
      is.close();
      log.info("Reading complete. Read {} vectors.", count);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return vectors;
  }

  public static void readBvecs(String filePath, int numRows, List<float[]> vectors) {
    log.info("Reading {} from file: {}", numRows, filePath);

    try {
      InputStream is = null;

      if (filePath.endsWith("bvecs.gz")) {
        is = new GZIPInputStream(new FileInputStream(filePath));
      } else if (filePath.endsWith(".bvecs")) {
        is = new FileInputStream(filePath);
      }

      int dimension = getDimension(is);

      float[] row = new float[dimension];
      int count = 0;
      int rc = 0;

      while (is.available() != 0) {

        ByteBuffer bbf = ByteBuffer.wrap(is.readNBytes(1));
        bbf.order(ByteOrder.LITTLE_ENDIAN);
        row[rc++] = bbf.get() & 0xff;

        if (rc == dimension) {
          vectors.add(row);
          count += 1;
          rc = 0;
          row = new float[dimension];

          // Skip last 4 bytes.
          is.readNBytes(4);

          if (count % 1000 == 0) {
            System.out.print(".");
          }

          if (numRows != -1 && count == numRows) {
            break;
          }
        }
      }
      System.out.println();
      is.close();
      log.info("Reading complete. Read {} vectors.", count);
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  // New method to read .fbin files (format: num_vectors, dimension, then vector data)
  // Corrected readFbin method for Wiki-88M .fbin files
  public static void readFbin(String filePath, int numRows, List<float[]> vectors) {
    log.info("Reading {} from file: {}", numRows, filePath);

    try (InputStream is = new FileInputStream(filePath)) {
      // Read num_vectors (first 4 bytes, little endian)
      byte[] numVecBytes = is.readNBytes(4);
      ByteBuffer numVecBuffer = ByteBuffer.wrap(numVecBytes).order(ByteOrder.LITTLE_ENDIAN);
      int numVectors = numVecBuffer.getInt();

      // Read dimension (next 4 bytes, little endian)
      byte[] dimBytes = is.readNBytes(4);
      ByteBuffer dimBuffer = ByteBuffer.wrap(dimBytes).order(ByteOrder.LITTLE_ENDIAN);
      int dimension = dimBuffer.getInt();

      log.info("File header - total vectors: {}, dimension: {}", numVectors, dimension);

      float[] row = new float[dimension];
      int count = 0;
      // Split raw sequential read (DISK) from float[] materialization (CPU) so we can see how
      // much of dataset-load is actually disk I/O vs per-vector parse/allocate overhead.
      long rawReadNanos = 0;
      long materializeNanos = 0;

      while (is.available() != 0) {
        long tRead = System.nanoTime();
        byte[] vectorBytes = is.readNBytes(dimension * 4);
        rawReadNanos += System.nanoTime() - tRead;
        if (vectorBytes.length != dimension * 4) break;
        long tMat = System.nanoTime();
        ByteBuffer bb = ByteBuffer.wrap(vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < dimension; i++) row[i] = bb.getFloat();
        vectors.add(row.clone());
        materializeNanos += System.nanoTime() - tMat;
        count++;
        if (numRows != -1 && count == numRows) break;
        if (count % 1000 == 0) System.out.print(".");
      }
      System.out.println();
      StageTimers.record(
          "dataset-read-raw [DISK]", rawReadNanos, (long) count * dimension * Float.BYTES);
      StageTimers.record("dataset-materialize [CPU]", materializeNanos);
      log.info("Reading complete. Read {} vectors out of {} in file.", count, numVectors);
    } catch (Exception e) {
      log.error("Error reading fbin file", e);
    }
  }

  // Fixed method to read .ibin files (ground truth neighbors)
  public static ArrayList<int[]> readIbin(String filePath, int numRows) {
    log.info("Reading {} from file: {}", numRows, filePath);
    ArrayList<int[]> vectors = new ArrayList<int[]>();

    try {
      InputStream is = new FileInputStream(filePath);

      // For .ibin ground truth files: Read num_vectors first, then dimension
      byte[] numVecBytes = is.readNBytes(4);
      ByteBuffer numVecBuffer = ByteBuffer.wrap(numVecBytes).order(ByteOrder.LITTLE_ENDIAN);
      int numVectors = numVecBuffer.getInt();

      byte[] dimBytes = is.readNBytes(4);
      ByteBuffer dimBuffer = ByteBuffer.wrap(dimBytes).order(ByteOrder.LITTLE_ENDIAN);
      int dimension = dimBuffer.getInt();

      log.info("Ground truth file - total vectors: {}, dimension: {}", numVectors, dimension);

      int count = 0;
      while (is.available() != 0 && (numRows == -1 || count < numRows)) {
        // Read dimension * 4 bytes (int values)
        byte[] vectorBytes = is.readNBytes(dimension * 4);
        if (vectorBytes.length != dimension * 4) {
          break; // End of file
        }

        ByteBuffer bb = ByteBuffer.wrap(vectorBytes);
        bb.order(ByteOrder.LITTLE_ENDIAN);

        int[] row = new int[dimension];
        for (int i = 0; i < dimension; i++) {
          row[i] = bb.getInt();
        }

        vectors.add(row);
        count++;

        if (count % 1000 == 0) {
          System.out.print(".");
        }
      }
      System.out.println();
      is.close();
      log.info("Reading complete. Read {} vectors out of {} total.", count, numVectors);
    } catch (Exception e) {
      e.printStackTrace();
    }
    return vectors;
  }
}
