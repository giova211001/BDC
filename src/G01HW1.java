import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import org.codehaus.janino.Java;
import scala.Char;
import scala.Tuple2;
import java.util.Locale;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

/**
 * Specifically, for the homework you must do the following tasks.
 *
 * 1) Write a method/function MRComputeStandardObjective that takes in input an RDD of (point,group) pairs (representing a set 𝑈=𝐴∪𝐵
 * ), and a set 𝐶
 *  of centroids, and returns the value of the objective function Δ(𝑈,𝐶)
 *  described above, thus ignoring the demopgraphic groups.
 *
 * 2) Write a method/function MRComputeFairObjective that takes in input an RDD of (point,group) pairs (representing points of a set 𝑈=𝐴∪𝐵
 * ), and a set 𝐶
 *  of centroids, and returns the value of the objective function Φ(𝐴,𝐵,𝐶)
 *  described above.
 *
 * 3) Write a method/function MRPrintStatistics that takes in input an RDD of (point,group) pairs (representing points of a set 𝑈=𝐴∪𝐵
 * ), and a set 𝐶
 *  of centroids, and computes and prints the triplets (𝑐𝑖,𝑁𝐴𝑖,𝑁𝐵𝑖)
 * , for 1≤𝑖≤𝐾=|𝐶|
 * , where 𝑐𝑖
 *  is the 𝑖
 * -th centroid in 𝐶
 * , and 𝑁𝐴𝑖,𝑁𝐵𝑖
 *  are the numbers of points of 𝐴
 *  and 𝐵
 * , respectively, in the cluster 𝑈𝑖
 *  centered in 𝑐𝑖
 * .
 *
 * 4) Write a program GxxHW1.java (for Java users) or GxxHW1.py (for Python users), where xx is your 2-digit group number (e.g., 04 or 25), which receives in input, as command-line arguments, a path to the file storing the input points, and 3 integers 𝐿,𝐾,𝑀
 * , and does the following:
 *
 * Prints the command-line arguments and stores  𝐿,𝐾,𝑀
 *  into suitable variables.
 * Reads the input points into an RDD of (point,group) pairs -called inputPoints-, subdivided into 𝐿
 *  partitions.
 * Prints the number 𝑁
 *  of points, the number 𝑁𝐴
 *  of points of group A, and the number 𝑁𝐵
 *  of points of group B (hence, 𝑁=𝑁𝐴+𝑁𝐵
 * ).
 * Computes a set 𝐶
 *  of 𝐾
 *  centroids by using the Spark implementation of the standard Lloyd's algorithm for the input points, disregarding the points' demographic groups, and using 𝑀
 *  as number of iterations.
 * Prints the values of the two objective functions Δ(𝑈,𝐶)
 *  and Φ(𝐴,𝐵,𝐶)
 * , computed by running  MRComputeStandardObjective and MRComputeFairObjective, respectively.
 * Runs MRPrintStatistics.
 */



public class G01HW1 {
    public static void main(String[] args) throws IOException {

        /*
        Set the location to Locale.US to have the output format with "." insthead of ","
        This ensures the output uses the period as decimal separator insthead of the comma
        (for example, 0.00123 insthead of 0,00123)
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
        int L = Integer.parseInt(args[1]);
        int K = Integer.parseInt(args[2]);
        int M = Integer.parseInt(args[3]);

        System.out.println("Input file = " + file_path + ", L = " + L + ", K = " + K + ", M = " + M);

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
        JavaRDD<String> raw_data = ctx.textFile(file_path).repartition(L).cache();

        // Setting the GLOBAL VARIABLES
        long points, number_a, number_b;
        points = raw_data.count(); //stampa N
        number_a = raw_data.filter(line -> line.trim().endsWith("A")).count(); //stampa NA
        number_b = raw_data.filter(line -> line.trim().endsWith("B")).count(); //stampa NB
        System.out.println("N = " + points + ", NA = " + number_a + ", NB = " + number_b);


        //  MAP - PHASE
        // Leggere il file e trasformarlo in Tuple2<Vector, Character>
        JavaPairRDD<Vector, Character> U = ctx.textFile(file_path).mapToPair(line -> {
            String[] parts = line.split(",");
            double[] values = {Double.parseDouble(parts[0]), Double.parseDouble(parts[1])}; // Coordinate
            Vector point = Vectors.dense(values);
            char label = parts[2].trim().charAt(0); // Etichetta A/B
            return new Tuple2<>(point, label);
        }).cache();

        // Estract only vector so compute the k-centroids
        JavaRDD<Vector> pointsRDD = U.keys();

        //Apply the standard method for compute the KMean
        KMeansModel model = KMeans.train(pointsRDD.rdd(), K, M);

        Vector[] centroids = model.clusterCenters();



        // Predire i cluster per ciascun punto
        //JavaRDD<Integer> clusterIndices = model.predict(pointsRDD);
        //clusterIndices.foreach(point -> System.out.println(point));

        // Supponendo che 'pointsRDD' sia un JavaRDD<Vector> contenente i punti
        // e che 'centroids' sia un array di Vector rappresentante i centroidi trovati con KMeans

        JavaPairRDD<Vector, Integer> pointsWithClosestCentroid = pointsRDD.mapToPair(point -> {
            int closest = findClosestCentroid(point, centroids);
            return new Tuple2<>(point, closest); // Restituisce la coppia (punto, centroide più vicino)
        });

        // Stampare i risultati
        //pointsWithClosestCentroid.foreach(tuple -> {
        //    System.out.println("Punto: " + tuple._1() + " --> Centroide più vicino: " + tuple._2());
        //});

        // Otteniamo un RDD contenente solo gli indici dei cluster
        JavaRDD<Integer> clusterAssignments = pointsWithClosestCentroid.map(Tuple2::_2);

        // Conta quanti punti appartengono a ciascun cluster
        Map<Integer, Long> clusterCounts = clusterAssignments.countByValue();

        // Stampiamo il conteggio dei punti per ogni cluster
        clusterCounts.forEach((cluster, count) ->
                System.out.println("Cluster " + cluster + ": " + count + " punti"));

        double standard = MRComputeStandardObjective(U, centroids);
        System.out.printf("Delta(U,C) = %.6f%n", standard);
        double fair = MRComputeFairObjective(U, centroids);
        System.out.printf("Phi(A,B,C) = %.6f%n", fair);

        MRPrintStatistics(U, centroids);


    }


    public static int findClosestCentroid(Vector point, Vector[] centroids)
    {
        Vector closest = null;
        // variable to save the minimum distance between the point and the nearest centroid
        double min_distance = Double.MAX_VALUE;
        int closest_idx = -1;

        //Scan all the centroids
        for(int i = 0; i < centroids.length; i++)
        {
            //Compute the distance between the point and the actual centroid
            double distance = Vectors.sqdist(point, centroids[i]);

            //Check if the distance calculate is less than min_distance
            if(distance < min_distance)
            {
                min_distance = distance;
                closest = centroids[i];
                closest_idx = i;

            }
        }
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
        //Every element is a pair ((x1,x2,.....,xD), group)
        //MAP PHASE
        // Assign every point at the nearest centroid
        JavaPairRDD<Integer, Character> assignment = all_points.mapToPair(p -> {
            Vector point = p._1;
            Character group = p._2;
            int cluster_idx = findClosestCentroid(point, centroids);
            return new Tuple2<>(cluster_idx, group);
        });

        //MAP PHASE
        // Convert into (key, value = 1) to count NA and NB
        JavaPairRDD<Tuple2<Integer, Character>, Integer> toCount = assignment.mapToPair( t -> {
            return new Tuple2<>(t, 1);
        }).reduceByKey((a,b) -> a + b); // REDUCE PHASE

        // Group and format the output
        Map<Tuple2<Integer, Character>, Integer> localMap = toCount.collectAsMap();
        // Crea una mappa per tenere traccia dei contatori NA e NB per ogni indice
        int[] NA = new int[centroids.length];  // contatori per A
        int[] NB = new int[centroids.length];  // contatori per B

        for (Map.Entry<Tuple2<Integer, Character>, Integer> entry : localMap.entrySet()) {
            Tuple2<Integer, Character> key = entry.getKey();
            Integer value = entry.getValue();
            // Assegna i contatori a NA o NB in base al carattere
            if (key._2() == 'A') {
                NA[key._1()] += value;
            } else if (key._2() == 'B') {
                NB[key._1()] += value;
            }
        }

        //Print result in the correct form
        for(int i = 0; i < centroids.length; i++)
        {
            Vector centroid = centroids[i];
            double x = centroid.apply(0);
            double y = centroid.apply(1);

            // Stampa formattata come richiesto
            System.out.printf("i = %d, center = (%.6f, %.6f), NA%d = %d, NB%d = %d%n",
                    i, x, y, i, NA[i], i, NB[i]);
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
        //MAP PHASE
        //Compute the squared distances
        JavaPairRDD<Integer, Double> distances = all_points.mapToPair( p -> {
            Vector point = p._1;
            int closest_idx_centroid = findClosestCentroid(point, centroids);
            double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);
            return new Tuple2<>(closest_idx_centroid, minDistance);
        });

        //REDUCE PHASE
        // Sum all the distances for every centroid
        JavaPairRDD<Integer, Double> totalDistances = distances.reduceByKey((a,b) -> a + b);
        // Compute the average
        long totalPoints = all_points.count();
        double sumDistance = totalDistances.map( t -> t._2).reduce((a,b) -> a + b);
        //return the value
        double delta = sumDistance / totalPoints;
        return delta;
    }

    public static double MRComputeFairObjective(JavaPairRDD<Vector, Character> all_points, Vector[] centroids)
    {
        //MAP PHASE
        JavaPairRDD<Tuple2<Integer, Character>, Double> distanceGroup = all_points.mapToPair( p -> {
            Vector point = p._1;
            Character group = p._2;
            int closest_idx_centroid = findClosestCentroid(point, centroids);
            double minDistance = Vectors.sqdist(point, centroids[closest_idx_centroid]);
            return new Tuple2<>(new Tuple2<>(closest_idx_centroid, group), minDistance);
        }).reduceByKey((a,b) -> a + b); //REDUCE PHASE

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


