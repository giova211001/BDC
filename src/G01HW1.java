import com.sun.istack.logging.Logger;
import org.apache.log4j.Level;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

import java.io.IOException;
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
        JavaRDD<String> input = ctx.textFile(file_path).repartition(L).cache();

        // Setting the GLOBAL VARIABLES
        long points, number_a, number_b;
        points = input.count();
        number_a = input.filter(line -> line.trim().endsWith("A")).count();
        number_b = input.filter(line -> line.trim().endsWith("B")).count();
        System.out.println("N = " + points + ", NA = " + number_a + ", NB = " + number_b);

        //Another version to count the A and the B
        Map<String, Long> counts = input
                .map(line -> line.trim().substring(line.trim().length() - 1)) // Take last character
                .filter(letter -> letter.equals("A") || letter.equals("B")) // Consider only "A" and "B" in the last char
                .countByValue(); // Count the occurences


        System.out.println("NA: " + counts.getOrDefault("A", 0L));
        System.out.println("NB: " + counts.getOrDefault("B", 0L));



        JavaPairRDD<Tuple2<Double, Double>, String> pairs;
        // Convert the input into pairs of the form (point, group)





    }
}

