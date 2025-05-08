import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.io.Serializable;
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


public class G01HW2 {

    public class ClusterStats {
        double alpha;
        double beta;
        Vector muA;
        Vector muB;
        double l;
    }

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
        SparkConf conf = new SparkConf(true).setAppName("G01HW2"); // Set the application name for Spark
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
            //CORRECTION FROM HW1 - assume that the point has N dimensions
            String[] parts = line.split(",");

            //All element except the last one are coordinates of the point
            int dimensions = parts.length - 1;
            double[] values = new double[dimensions];
            for(int i = 0; i < dimensions; i++)
            {
                values[i] = Double.parseDouble(parts[i]);
            }

            Vector point = Vectors.dense(values); // Create Vector for the point with the dense() method
            char label = parts[parts.length - 1].trim().charAt(0); // Extract the label ('A' or 'B')
            return new Tuple2<>(point, label);
        }).cache(); // Cache the RDD for performance


        System.out.println("COMPUTAZIONE STANDARD");
        long startStandard = System.currentTimeMillis(); // Inizio
        List<Vector> C_stand = MRLloyd(U, K, M);
        long endStandard = System.currentTimeMillis();   // Fine
        // Calcolo tempo totale in millisecondi
        long elapsedStandard = endStandard - startStandard;
        // Conversione in secondi
        double elapsedSecondsS = elapsedStandard / 1000.0;

        System.out.println("Tempo di esecuzione: " + elapsedSecondsS + " secondi");
        System.out.println("OBJECTIVE FUNCTION STANDARD " + MRComputeFairObjective(U, C_stand.toArray(new Vector[0])));

        //COMPUTAZIONE FATTA DA NOI
        System.out.println("COMPUTAZIONE FAIR");
        long startFair = System.currentTimeMillis();
        List<Vector> C_Fair = MRFairLloyd(U, K, M);
        long endFair = System.currentTimeMillis();
        long elapsedFair = endFair - startFair;
        double elapsedSecondsF = elapsedFair/ 1000.0;
        System.out.println("Tempo di esecuzione: " + elapsedSecondsF + " secondi");
        System.out.println("OBJECTIVE FUNCTION STANDARD " + MRComputeFairObjective(U, C_Fair.toArray(new Vector[0])));


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


    //function implement by the professor to compute the new set of centroids
    public static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell, int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t=1; t<=T; t++){
            fA = fixedA;
            fB = fixedB;
            power = power/2;
            for (int i=0; i<K; i++) {
                double temp = (1-gamma)*beta[i]*ell[i]/(gamma*alpha[i]+(1-gamma)*beta[i]);
                xDist[i]=temp;
                fA += alpha[i]*temp*temp;
                temp=(ell[i]-temp);
                fB += beta[i]*temp*temp;
            }
            if (fA == fB) {break;}
            gamma = (fA > fB) ? gamma+power : gamma-power;
        }
        return xDist;
    }

    //Default method offered by Spark to compute the centroids with K clusters and M iterations
    public static List<Vector> MRLloyd(JavaPairRDD<Vector, Character> all_points, int K, int M)
    {
        //Inizialize a set C of K centroids
        // Estract only the point (vector) from the (point, group) pairs to compute the k-centroids
        JavaRDD<Vector> pointsRDD = all_points.keys();

        // Apply KMeans to compute the centroids (cluster centers)
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, M); // Train the KMeans model with K clusters and M iterations
        Vector[] centroids = model.clusterCenters();
        List<Vector> centroidList = Arrays.asList(centroids);
        return centroidList;
    }


    //Method implements by us to find the centroids
    public static List<Vector> MRFairLloyd(JavaPairRDD<Vector, Character> all_points, int K, int M)
    {
        // Step 1: inizializza K centroidi con KMeans|| (Spark)
        // Step 2: per M volte:
        //         a. assegna ogni punto al centroide più vicino
        //         b. per ogni gruppo di punti (cluster), calcola un nuovo centroide "fair"
        // Step 3: ritorna la lista dei centroidi finali

        //Inizialize a set C of K centroids
        // Estract only the point (vector) from the (point, group) pairs to compute the k-centroids
        JavaRDD<Vector> pointsRDD = all_points.keys();

        // Apply KMeans to compute the centroids (cluster centers)
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, 0); // Train the KMeans model with K clusters and M iterations
        Vector[] centroids = model.clusterCenters();
        List<Vector> centroidList = Arrays.asList(centroids);

        for (int iter = 0; iter < M; iter++) {
            Broadcast<Vector[]> broadcastCentroids = JavaSparkContext.fromSparkContext(all_points.context())
                    .broadcast(centroidList.toArray(new Vector[0]));

            // Assegna ogni punto al centroide più vicino
            JavaPairRDD<Integer, Tuple2<Vector, Character>> clusteredPoints = all_points.mapToPair(point -> {
                Vector vec = point._1;
                Character group = point._2;
                int closest = findClosestCentroid(vec, broadcastCentroids.value());
                return new Tuple2<>(closest, new Tuple2<>(vec, group));
            });

            // Raggruppa per cluster
            JavaPairRDD<Integer, Iterable<Tuple2<Vector, Character>>> groupedByCluster = clusteredPoints.groupByKey();

            // Conta i totali globali per A e B (li calcoliamo una sola volta fuori dal ciclo)
            long totalA = all_points.filter(p -> p._2 == 'A').count();
            long totalB = all_points.filter(p -> p._2 == 'B').count();

            // Inizializza i vettori
            double[] alpha = new double[K];
            double[] beta = new double[K];
            double[] l = new double[K];
            Vector[] muA = new Vector[K];
            Vector[] muB = new Vector[K];

            Map<Integer, Iterable<Tuple2<Vector, Character>>> clusterMap = groupedByCluster.collectAsMap();

            for (int i = 0; i < K; i++) {
                Iterable<Tuple2<Vector, Character>> points = clusterMap.get(i);
                List<Vector> groupA = new ArrayList<>();
                List<Vector> groupB = new ArrayList<>();

                for (Tuple2<Vector, Character> entry : points) {
                    if (entry._2 == 'A') groupA.add(entry._1);
                    else if (entry._2 == 'B') groupB.add(entry._1);
                }

                muA[i] = computeCentroid(groupA);
                muB[i] = computeCentroid(groupB);

                alpha[i] = (double) groupA.size() / totalA;
                beta[i] = (double) groupB.size() / totalB;
                l[i] = Math.sqrt(Vectors.sqdist(muA[i], muB[i]));
            }

            // Calcola fixedA
            double deltaA = all_points.filter(p -> p._2 == 'A').mapToDouble(p -> {
                int cluster = findClosestCentroid(p._1, broadcastCentroids.value());
                return squaredDistance(p._1, muA[cluster]);
            }).reduce(Double::sum);

            // Calcola fixedB
            double deltaB = all_points.filter(p -> p._2 == 'B').mapToDouble(p -> {
                int cluster = findClosestCentroid(p._1, broadcastCentroids.value());
                return squaredDistance(p._1, muB[cluster]);
            }).reduce(Double::sum);

            double fixedA = deltaA / totalA;
            double fixedB = deltaB / totalB;

            // Calcola x tramite la funzione fornita
            double[] x = computeVectorX(fixedA, fixedB, alpha, beta, l, K);

            // Calcola i nuovi centroidi fair
            List<Vector> newCentroids = new ArrayList<>();
            for (int i = 0; i < K; i++) {
                Vector ci = combineVectors(muA[i], muB[i], (l[i] - x[i]) / l[i], x[i] / l[i]);
                newCentroids.add(ci);
            }

            centroidList = newCentroids;
        }


        return centroidList;
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

            //new code added to correct feedback from hw1
            Vector center = centroids[centroidId];
            StringBuilder coords = new StringBuilder();
            coords.append("(");
            for (int i = 0; i < center.size(); i++) {
                coords.append(String.format("%.6f", center.apply(i)));
                if (i < center.size() - 1)
                    coords.append(", ");
            }
            coords.append(")");

            System.out.printf("i = %d, center = %s, NA%d = %d, NB%d = %d%n",
                    centroidId, coords.toString(), centroidId, countA, centroidId, countB);

        }

    }
    public static Vector computeCentroid(List<Vector> vectors) {

        if (vectors.isEmpty()) return Vectors.zeros(0);

        int dim = vectors.get(0).size();
        double[] sum = new double[dim];

        for (Vector vec : vectors) {
            for (int i = 0; i < dim; i++) {
                sum[i] += vec.apply(i);
            }
        }

        for (int i = 0; i < dim; i++) {
            sum[i] /= vectors.size();
        }

        return Vectors.dense(sum);
    }

    public static double squaredDistance(Vector v1, Vector v2) {
        int dim = v1.size();
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
        double diff = v1.apply(i) - v2.apply(i);
        sum += diff * diff;
        }
        return sum;
        }

        public static Vector combineVectors(Vector v1, Vector v2, double weight1, double weight2) {
        int dim = v1.size();
        double[] result = new double[dim];
        for (int i = 0; i < dim; i++) {
            result[i] = weight1 * v1.apply(i) + weight2 * v2.apply(i);
        }
        return Vectors.dense(result);
    }



}










