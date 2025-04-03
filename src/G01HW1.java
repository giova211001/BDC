import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;
import scala.Char;
import scala.Tuple2;

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

        /*Check the number of CMD LINE PARAMETERS in order to satisfy the following requirement of the homework
         -> Prints the command-line arguments and stores  𝐿,𝐾,𝑀 into suitable variables.
         INPUTS:
         1) path to the file storing the input points
         2) L = number of partitions
         3) K = number of desired clusters
         4) M = number of iterations

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
            SPARK SETUP
         */

        SparkConf conf = new SparkConf(true).setAppName("G01HW1");
        JavaSparkContext ctx = new JavaSparkContext(conf);
        ctx.setLogLevel("OFF");

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

        //Another version to count the A and the B
        Map<String, Long> counts = raw_data
                .map(line -> line.trim().substring(line.trim().length() - 1)) // Take last character
                .filter(letter -> letter.equals("A") || letter.equals("B")) // Consider only "A" and "B" in the last char
                .countByValue(); // Count the occurences


        System.out.println("NA: " + counts.getOrDefault("A", 0L));
        System.out.println("NB: " + counts.getOrDefault("B", 0L));

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

        System.out.println("Delta(U,C):" + MRComputeStandardObjective(pointsRDD, centroids));

        //Stampa a schermo


        // Predire i cluster per ciascun punto
        //JavaRDD<Integer> clusterIndices = model.predict(pointsRDD);
        //clusterIndices.foreach(point -> System.out.println(point));

        // Supponendo che 'pointsRDD' sia un JavaRDD<Vector> contenente i punti
        // e che 'centroids' sia un array di Vector rappresentante i centroidi trovati con KMeans

        JavaPairRDD<Vector, Integer> pointsWithClosestCentroid = pointsRDD.mapToPair(point -> {
            Vector closest = findClosestCentroid(point, centroids);
            int index = Arrays.asList(centroids).indexOf(closest);
            return new Tuple2<>(point, index); // Restituisce la coppia (punto, centroide più vicino)
        });

        // Stampare i risultati
        pointsWithClosestCentroid.foreach(tuple -> {
            System.out.println("Punto: " + tuple._1() + " --> Centroide più vicino: " + tuple._2());
        });

        // Otteniamo un RDD contenente solo gli indici dei cluster
        JavaRDD<Integer> clusterAssignments = pointsWithClosestCentroid.map(Tuple2::_2);

        // Conta quanti punti appartengono a ciascun cluster
        Map<Integer, Long> clusterCounts = clusterAssignments.countByValue();

        // Stampiamo il conteggio dei punti per ogni cluster
        clusterCounts.forEach((cluster, count) ->
                System.out.println("Cluster " + cluster + ": " + count + " punti"));





    }


    public static Vector findClosestCentroid(Vector point, Vector[] centroids)
    {
        Vector closest = null;
        // variable to save the minimum distance between the point and the nearest centroid
        double min_distance = Double.MAX_VALUE;

        //Scan all the centroids
        for(Vector centroid: centroids)
        {
            //Compute the distance between the point and the actual centroid
            double distance = Vectors.sqdist(point, centroid);

            //Check if the distance calculate is less than min_distance
            if(distance < min_distance)
            {
                min_distance = distance;
                closest = centroid;
            }
        }
        return closest;

    }

    public static double MRComputeStandardObjective(JavaRDD<Vector> pointsRDD, Vector[] centroids) {
        // Numero totale di punti
        long numPoints = pointsRDD.count();

        // Calcoliamo la somma delle distanze quadrate tra ogni punto e il centroide più vicino
        double totalSquaredDistance = pointsRDD
                .map(point -> Vectors.sqdist(point, findClosestCentroid(point, centroids))) // Calcola d(u, C)^2
                .reduce(Double::sum); // Somma tutte le distanze



        // Evitiamo divisione per zero nel caso di input vuoto
        return (numPoints == 0) ? 0.0 : totalSquaredDistance / numPoints;
    }
}

