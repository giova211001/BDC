import org.apache.commons.math3.random.RandomGenerator;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.codehaus.janino.Java;
import scala.Char;
import scala.Int;
import scala.Tuple2;

import java.util.*;

import java.io.IOException;


/**



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


        /*
            SPARK SETUP : Initialize Spark context and configurations
         */

        SparkConf conf = new SparkConf(true).setAppName("G01HW1"); // Set the application name for Spark
        JavaSparkContext ctx = new JavaSparkContext(conf); // Create the JavaSparkContext to interact with Spark
        ctx.setLogLevel("OFF"); // Turn off Spark logging for cleaner output

        /*
            Store the input file into the RDD and subdivide into L partitions
            textFile method -> transform the input file into an RDD of Strings, whose element correspond to the
            distinct lines of thr file
         */

        // Print the parameters
        System.out.println("Input file = " + file_path + ", L = " + L + ", K = " + K + ", M = " + M);

        // Read the input file into an RDD and repartition it into L partitions
        JavaRDD<String> raw_data = ctx.textFile(file_path).repartition(L).cache();

        // Global variables to store counts of points, points in group A, and points in group B
        long points, number_a, number_b;
        points = raw_data.count(); // Total number of points
        number_a = raw_data.filter(line -> line.trim().endsWith("A")).count(); // Count of points of group A
        number_b = raw_data.filter(line -> line.trim().endsWith("B")).count(); // Count of points of group B
        System.out.println("N = " + points + ", NA = " + number_a + ", NB = " + number_b);


        // MAP PHASE: Transform the input data into a tuple of (point, group) pairs
        JavaPairRDD<Vector, Character> U = raw_data.mapToPair(line -> {
            String[] parts = line.split(",");
            double[] values = {Double.parseDouble(parts[0]), Double.parseDouble(parts[1])}; // // Extract point coordinates
            Vector point = Vectors.dense(values); // Create Vector for the point with the dense() method
            char label = parts[2].trim().charAt(0); // Extract the label ('A' or 'B')
            return new Tuple2<>(point, label);
        }).cache(); // Cache the RDD for performance reasons

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
     * @return The index of the closest centroid.
     */
    public static int findClosestCentroid(Vector point, Vector[] centroids)
    {
        // variable to save the minimum distance between the point and the nearest centroid
        double min_distance = Double.MAX_VALUE; // Initialize minimum distance to a very large value
        int closest_idx = -1; // Initialize the index of the closest centroid

        // Iterate through all centroids to find the closest one
        for(int i = 0; i < centroids.length; i++)
        {
            // Compute the squared Euclidean distance between the point and the current centroid
            // with the method sqdist of class Vector
            double distance = Vectors.sqdist(point, centroids[i]);

            //Check if the distance calculate is less than min_distance
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
     * Computes and prints, for each cluster, the number of points from each demographic group (A and B).
     * For each centroid ci, the method prints:
     * - its index and coordinates,
     * - the number of points from group A assigned to ci,
     * - the number of points from group B assigned to ci.
     *
     * @param all_points An RDD of pairs (point, group), where each point is represented as a Vector,
     *                   and the group is a Character label ('A' or 'B').
     * @param centroids An array of Vectors representing the final centroids computed by k-means.
     *
     */

    public static void MRPrintStatistics(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {


        JavaPairRDD<Integer, Tuple2<Integer, Integer>> points = all_points.mapToPair(p -> {
            Vector point = p._1;
            Character lab = p._2;

            int centroid = findClosestCentroid(point, centroids);
            if(lab == 'A') return new Tuple2<>(centroid, new Tuple2<>(1,0));
            else return new Tuple2<>(centroid, new Tuple2<>(0,1));
        }).reduceByKey((a,b) -> {
            int localA = a._1 + b._1;
            int localB = a._2 + b._2;

            return new Tuple2<>(localA, localB);
        }).sortByKey();

        List<Tuple2<Integer, Tuple2<Integer, Integer>>> result = points.collect();

        for (Tuple2<Integer, Tuple2<Integer, Integer>> entry : result) {
            int centroidId = entry._1;
            int countA = entry._2._1;
            int countB = entry._2._2;



            // Stampa formattata come richiesto
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
     * @return The value of Δ(U, C)
     *
     */
    public static double MRComputeStandardObjective(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {

        // MAP PHASE (per partizione): Calcola la distanza al quadrato di ogni punto dal centroide più vicino
        JavaPairRDD<Integer, Double> distances = all_points.mapPartitionsToPair(partition -> {
            List<Tuple2<Integer, Double>> results = new ArrayList<>();
            while (partition.hasNext()) {
                Tuple2<Vector, Character> p = partition.next();
                Vector point = p._1;
                int closest_idx_centroid = findClosestCentroid(point, centroids);
                double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);
                results.add(new Tuple2<>(closest_idx_centroid, minDistance));
            }
            return results.iterator();
        }).reduceByKey((a,b) -> a + b); // REDUCE PHASE: Sum the squared distances for each centroid


        //JavaPairRDD<Integer, Double> totalDistances = distances.reduceByKey((a,b) -> a + b);
        // Compute the total distance and the average distance
        long totalPoints = all_points.count();
        double sumDistance = distances.map( t -> t._2).reduce((a,b) -> a + b);
        // Return the average squared distance (Delta)
        double delta = sumDistance / totalPoints;
        return delta;
    }

    /**
     * Computes the fair objective function (Phi) that ensures fairness between the two demographic groups,
     * considering the distance from each group to their closest centroid.
     *
     * @param all_points An RDD of pairs (point, group), where point is a Vector and group is a Character ('A' or 'B').
     * @param centroids An array of Vectors representing the centroids of the clusters.
     * @return The value of the fair objective function Φ(A, B, C).
     */
    public static double MRComputeFairObjective(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {
        // MAP PHASE: Compute the squared distance from each point to its closest centroid and its group
        JavaPairRDD<Tuple2<Integer, Character>, Double> distanceGroup = all_points.mapPartitionsToPair(partition -> {
            List<Tuple2<Tuple2<Integer, Character>, Double>> results = new ArrayList<>();
            while (partition.hasNext()) {
                Tuple2<Vector, Character> p = partition.next();
                Vector point = p._1;
                Character group = p._2;
                int closest_idx_centroid = findClosestCentroid(point, centroids);
                double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);
                results.add(new Tuple2<>(new Tuple2<>(closest_idx_centroid, group), minDistance));
            }
            return results.iterator();
        }).reduceByKey((a, b) -> a + b);

        JavaPairRDD<Character, Double> distanceForGroup = distanceGroup.mapToPair( entry -> {
            Tuple2<Integer, Character> key = entry._1;
            double dist = entry._2;

            // I have to map to obtain (char, sum of distances)
            return new Tuple2<>(key._2, dist);
        }).reduceByKey((a,b) -> a + b);

        long NA = all_points.filter(p -> p._2 == 'A').count();
        long NB = all_points.filter(p -> p._2 == 'B').count();

        // Converting RDD to a Map to access values directly
        Map<Character, Double> groupDistancesMap = distanceForGroup.collectAsMap();
        double totalA = groupDistancesMap.getOrDefault('A', 0.0);
        double totalB = groupDistancesMap.getOrDefault('B', 0.0);

        double fairObjectiveA = totalA / NA;
        double fairObjectiveB = totalB / NB;

        double phi = Math.max(fairObjectiveA, fairObjectiveB);

        return phi;
    }

}







