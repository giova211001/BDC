import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.*;

import java.io.IOException;


/**
 *
 * This program performs clustering on a 2D dataset using the KMeans algorithm implemented in Apache Spark.
 * In addition to the standard k-means objective function, it computes a fairness-aware clustering metric
 * that accounts for demographic parity between two groups labeled 'A' and 'B'.
 *
 * The input consists of 2D points with a demographic label. The algorithm processes this input,
 * clusters the data into K groups using L partitions over M iterations, and computes:
 * - The standard k-means cost (Delta) --> function MRComputeStandardObjective
 * - The fairness k-means cost (Phi) --> function MRComputeFairObjective
 * - Statistics of group distribution across clusters --> function MRPrintStatistics
 *
 * Input Format (CSV): x, y, label (where label is either 'A' or 'B')
 *
 * Dependencies: Apache Spark, Spark MLlib
 *
 * Authors:
 * - Faedo Giovanni - Student ID: 2149759
 * - Prioli Giacomo - Student ID: 2166293
 * - Francescato Daniele - Student ID: 2160563
 */


public class G01HW1 {

    public static void main(String[] args) throws IOException {

        /*
        Set the location to Locale.US to have the output format with "." instead of ","
        This ensures the output uses the period as decimal separator instead of the comma
        (for example, 0.00123 instead of 0,00123)
         */
        Locale.setDefault(Locale.US);


        /*
        * Check the number of CMD LINE PARAMETERS to ensure the input meets the homework requirements:
        * The program expects 4 arguments:
         * 1) Path to the file storing the input points.
        * 2) L = number of partitions.
        * 3) K = number of desired clusters.
        * 4) M = number of iterations.
        */
        if(args.length != 4){
            throw new IllegalArgumentException("USAGE: file_path num_partitions num_cluster num_iterations");
        }


        // Store and print the COMMAND LINE ARGUMENT
        String file_path = args[0];
        int L = Integer.parseInt(args[1]); // Number of partitions
        int K = Integer.parseInt(args[2]); // Number of desired clusters
        int M = Integer.parseInt(args[3]); // Number of iterations for KMeans

        /**
         * Validates that K (number of clusters) and M (number of iterations) are positive integers.
         *
         * K-means cannot execute with:
         * - K ≤ 0 (at least 1 cluster is required)
         * - M ≤ 0 (at least 1 iteration is needed to compute centroids)
         *
         * @throws IllegalArgumentException if K or M are non-positive
         */

        if (K <= 0 || M <= 0) {
            throw new IllegalArgumentException("K and M must be positive integers.");
        }

        /**
         * Basic validation: file_path must be non-null and non-empty
         * @throws IllegalArgumentException if K or M are non-positive
         */
        if (file_path == null || file_path.trim().isEmpty()) {
            throw new IllegalArgumentException("File path cannot be null or empty.");
        }

        // SPARK SETUP : Initialize Spark context and configurations
        SparkConf conf = new SparkConf(true).setAppName("G01HW1"); // Set the application name for Spark
        JavaSparkContext ctx = new JavaSparkContext(conf); // Create the JavaSparkContext to interact with Spark
        ctx.setLogLevel("OFF"); // Turn off Spark logging for cleaner output

        /*
            Store the input file into the RDD and subdivide into L partitions
            textFile method -> transform the input file into an RDD of Strings, whose element correspond to the
            distinct lines of the file
         */
        // Read the input file into an RDD and repartition it into L partitions
        JavaRDD<String> inputPoints = ctx.textFile(file_path).repartition(L).cache();

        // Global variables to store counts of points, points in group A, and points in group B
        long N, NA, NB;
        N = inputPoints.count(); // Total number of points
        NA = inputPoints.filter(line -> line.trim().endsWith("A")).count(); // Count of points of group A
        NB = inputPoints.filter(line -> line.trim().endsWith("B")).count(); // Count of points of group B

        // Print input parameters and dataset statistic (total number of points, total number of points
        // with label 'A' and total number of points with label 'B')
        System.out.println("Input file = " + file_path + ", L = " + L + ", K = " + K + ", M = " + M);
        System.out.println("N = " + N + ", NA = " + NA + ", NB = " + NB);


        // MAP PHASE: Transform the input data into a tuple of (Vector, Label) pairs
        JavaPairRDD<Vector, Character> U = inputPoints.mapToPair(line -> {
            String[] parts = line.split(",");
            double[] values = {Double.parseDouble(parts[0]), Double.parseDouble(parts[1])}; // // Extract point coordinates
            Vector point = Vectors.dense(values); // Create Vector for the point with the dense() method
            char label = parts[2].trim().charAt(0); // Extract the label ('A' or 'B')
            return new Tuple2<>(point, label);
        }).cache(); // Cache the RDD for performance

        // Estract only the point (vector) from the (point, group) pairs to compute the k-centroids
        JavaRDD<Vector> pointsRDD = U.keys();

        // Apply KMeans to compute the centroids (cluster centers)
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, M); // Train the KMeans model with K clusters and M iterations
        Vector[] centroids = model.clusterCenters();

        // Compute the standard objective function (Delta)
        double standard = MRComputeStandardObjective(U, centroids);
        System.out.printf("Delta(U,C) = %.6f%n", standard);

        // Compute the fair objective function (Phi)
        double fair = MRComputeFairObjective(U, centroids);
        System.out.printf("Phi(A,B,C) = %.6f%n", fair);

        // Print the statistics (cluster assignments and counts of group A and B points)
        MRPrintStatistics(U, centroids);

        // Close the Spark context
        ctx.close();
    }

    /**
     * Finds the closest centroid to a given point by computing the squared Euclidean distance.
     *
     * @param point The point whose closest centroid is to be found.
     * @param centroids The array of centroids to which the point will be compared.
     * @throws IllegalArgumentException if centroids array is empty.
     * @return The index of the closest centroid.
     */
    public static int findClosestCentroid(Vector point, Vector[] centroids)
    {
        if(centroids.length == 0) {
            throw new IllegalArgumentException("Centroids array cannot be empty.");
        }

        // variable to save the minimum distance between the point and the nearest centroid
        double min_distance = Double.MAX_VALUE; // Initialize minimum distance to a very large value
        int closest_idx = -1; // Initialize the index of the closest centroid

        // Iterate through all centroids to find the closest one
        for(int i = 0; i < centroids.length; i++)
        {
            // Compute the squared Euclidean distance between the point and the current centroid
            // with the method sqdist of class Vector
            double distance = Vectors.sqdist(point, centroids[i]);

            // Update closest centroid if a nearer one is found
            if(distance < min_distance)
            {
                min_distance = distance;
                closest_idx = i;

            }
        }

        // Return the index of the closest centroid
        return closest_idx;

    }

    /**
     * Computes and prints statistics for each cluster formed by the KMeans algorithm.
     *
     * This method calculates, for each centroid, how many points of group A and group B
     * have been assigned to it. The assignment is based on the closest centroid to each point
     * using the squared Euclidean distance. The statistics are then printed in a formatted
     * output for each cluster.
     *
     * @param all_points A JavaPairRDD where each element is a tuple containing a point (as a Vector)
     *                   and its associated demographic group ('A' or 'B').
     * @param centroids  An array of cluster centroids computed by the KMeans algorithm.
     * @throws IllegalArgumentException if centroids array is empty.
     */
    public static void MRPrintStatistics(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {

        if(centroids.length == 0) {
            throw new IllegalArgumentException("Centroids array cannot be empty.");
        }

        /*
         * MAP PHASE (per partition):
         * For each point in the partition, find its closest centroid.
         * Then emit a pair (centroid index, (1, 0)) if the point belongs to group A,
         * or (centroid index, (0, 1)) if it belongs to group B.
         * The resulting RDD has structure: (centroid index, (countA, countB))
         */
        JavaPairRDD<Integer, Tuple2<Integer, Integer>> points = all_points.mapPartitionsToPair(iter -> {

            // Temporary list to collect output tuples from this partition
            List<Tuple2<Integer, Tuple2<Integer, Integer>>> results = new ArrayList<>();

            // Iterate over all points in the partition
            while(iter.hasNext()){
                Tuple2<Vector, Character> p = iter.next();
                Vector point = p._1;
                Character lab = p._2;

                // Find the index of the closest centroid for the current point
                int centroid_idx = findClosestCentroid(point, centroids);


                // Create a pair (centroid index, (1, 0)) if the point belongs to group A,
                // or (centroid index, (0, 1)) if it belongs to group B
                if (lab == 'A')
                    results.add(new Tuple2<>(centroid_idx, new Tuple2<>(1, 0)));
                else
                    results.add(new Tuple2<>(centroid_idx, new Tuple2<>(0, 1)));
            }

            // Return all results for this partition
            return results.iterator();


        })

        /*
        * REDUCE PHASE:
        * Sum the (countA, countB) tuples for each centroid index to obtain the total
        * number of A and B points assigned to each cluster.
        */

        .reduceByKey((a,b) -> {
            int localA = a._1 + b._1; // Sum of points with label A
            int localB = a._2 + b._2; // Sum of points with label B

            return new Tuple2<>(localA, localB); // Return updated count pair
        })

        // Sort the results by centroid index (for clean output order)
        .sortByKey();

        /*
         * Print formatted output for each centroid.
         * For each centroid index, print its coordinates along with the number of A and B points assigned to it.
         * The output format is:
         * i = <centroid index>, center = (<x>, <y>), NA<i> = <count>, NB<i> = <count>
         */

        List<Tuple2<Integer, Tuple2<Integer, Integer>>> result = points.collect();
        for (Tuple2<Integer, Tuple2<Integer, Integer>> entry : result) {
            int centroidId = entry._1;
            int countA = entry._2._1;
            int countB = entry._2._2;

            // Print centroid index, coordinates, and the number of A/B points assigned
            System.out.printf("i = %d, center = (%.6f, %.6f), NA%d = %d, NB%d = %d%n",
                    centroidId, centroids[centroidId].apply(0), centroids[centroidId].apply(1), centroidId, countA, centroidId, countB);
        }

    }


    /**
     * Computes the standard k-means clustering objective function Δ(U, C),
     * defined as the average squared Euclidean distance from each point in the dataset
     * to its closest centroid. This function does not take demographic group information into account.
     *
     * @param all_points An RDD of pairs (point, group), where point is a Vector
     *  *                and group is a Character ('A' or 'B').
     * @param centroids An array of Vectors representing the set of cluster centroids computed by the
     *                  function KMean.train().
     * @throws IllegalArgumentException if centroids array is empty.
     * @return The value of Δ(U, C)
     *
     */
    public static double MRComputeStandardObjective(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {

        if(centroids.length == 0) {
            throw new IllegalArgumentException("Centroids array cannot be empty.");
        }

        /*
         * MAP PHASE (per partition):
         * For each point, find the closest centroid and compute the squared Euclidean distance
         * between the point and that centroid. The output is a pair:
         * (centroid index, squared distance)
         */

        JavaPairRDD<Integer, Double> distances = all_points.mapPartitionsToPair(partition -> {
            // Temporary list to collect (centroid index, squared distance) pairs
            List<Tuple2<Integer, Double>> results = new ArrayList<>();

            // Iterate through each point in the partition
            while (partition.hasNext()) {
                Tuple2<Vector, Character> p = partition.next();
                Vector point = p._1;

                // Find the index of the closest centroid for the current point
                int closest_idx_centroid = findClosestCentroid(point, centroids);

                // Compute the squared distance from the point to the closest centroid
                double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);

                // Add the result pair to the list
                results.add(new Tuple2<>(closest_idx_centroid, minDistance));
            }

            // Return an iterator over the list of results
            return results.iterator();
        })

        /*
        * REDUCE PHASE:
        * Sum all squared distances associated with the same centroid index.
        * This yields the total squared distance per centroid.
        */

        .reduceByKey((a, b) -> a + b); // REDUCE PHASE: Sum the squared distances for each centroid


        // Count the total number of points in the dataset
        long totalPoints = all_points.count();

        // Sum all squared distances across all centroids
        double sumDistance = distances.map( t -> t._2).reduce((a, b) -> a + b);

        // Compute the average squared distance (delta value)
        double delta = sumDistance / totalPoints;

        return delta;
    }

    /**
     * Computes the fair objective function (Phi) that ensures fairness between the two demographic groups,
     * considering the distance from each group to their closest centroid.
     *
     * @param all_points An RDD of pairs (point, group), where point is a Vector and group is a Character ('A' or 'B').
     * @param centroids An array of Vectors representing the centroids of the clusters.
     * @throws IllegalArgumentException if centroids array is empty.
     * @throws IllegalStateException if one group has no points, making fairness computation impossible.
     * @return The value of the fair objective function Φ(A, B, C).
     */
    public static double MRComputeFairObjective(JavaPairRDD<Vector, Character> all_points, Vector[] centroids) {

        if(centroids.length == 0) {
            throw new IllegalArgumentException("Centroids array cannot be empty.");
        }

        /*
         * MAP PHASE (per partition):
         * For each point in the dataset:
         * - Compute the squared distance to its closest centroid.
         * - Emit a pair (group label, squared distance).
         * The result is a collection of (A, dist) and (B, dist) pairs.
         */
        JavaPairRDD<Character, Double> mapped = all_points.mapPartitionsToPair(partition -> {

            // Temporary list to hold intermediate (group, distance) pairs
            List<Tuple2<Character, Double>> results = new ArrayList<>();

            // Iterate over each point in the partition
            while (partition.hasNext()) {
                Tuple2<Vector, Character> p = partition.next();
                Vector point = p._1;
                Character group = p._2;

                // Find the closest centroid and compute the squared distance
                int closest_idx_centroid = findClosestCentroid(point, centroids);
                double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);

                // Add the result pair to the list
                results.add(new Tuple2<>(group, minDistance));
            }
            // Return iterator over the results
            return results.iterator();
        })

        /*
        * REDUCE PHASE:
        * Sum the squared distances for each group label ('A' and 'B').
        * Resulting RDD: ('A', totalDistanceA), ('B', totalDistanceB)
        */

        .reduceByKey((a, b) -> a + b);


        // Count the number of points in each group to compute averages
        long NA = all_points.filter(p -> p._2 == 'A').count();
        long NB = all_points.filter(p -> p._2 == 'B').count();


        // Ensure neither group A nor B is empty to avoid division by zero
        if (NA == 0 || NB == 0) {
            throw new IllegalStateException("Fairness cannot be computed: one group has no points. " +
                    "NA = " + NA + ", NB = " + NB);
        }

        // Collect the results into a Map for printing
        Map<Character, Double> result = mapped.collectAsMap();

        /*
         * Collect the reduced results into a local map:
         * result.get('A') → total squared distance for group A
         * result.get('B') → total squared distance for group B
         * (default to 0.0 if a group is missing)
         */
        double totalA = result.getOrDefault('A', 0.0);
        double totalB = result.getOrDefault('B', 0.0);

        // Compute the average squared distance for each group
        double fairObjectiveA = totalA / NA;
        double fairObjectiveB = totalB / NB;

        // Return the maximum of the two group costs (phi value)
        double phi = Math.max(fairObjectiveA, fairObjectiveB);

        return phi;


    }

}







