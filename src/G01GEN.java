import java.io.PrintWriter;
import java.util.Locale;
import java.util.Random;

/**
 * G01GEN.java
 * Generates a synthetic dataset with N points in R^2 and assigns them to demographic groups A or B.
 * The dataset is structured to show a difference in clustering quality between standard and fair k-means.
 *
 * Usage: java G01GEN N K > dataset.csv
 */
public class G01GEN {
    public static void main(String[] args) {
        Locale.setDefault(Locale.US); // Ensure '.' as decimal separator

        // Validation of the input arguments
        if (args.length != 2) {
            System.err.println("USAGE: java G01GEN N K");
            System.exit(1);
        }

        int N = Integer.parseInt(args[0]);  // Total number of points
        int K = Integer.parseInt(args[1]);  // Number of desired clusters

        // Balanced distribution of the points
        int pointsPerCluster = N / (2 * K); // Half for each group, evenly spread across K clusters
        int generated = 0;
        Random rand = new Random(42); // Fixed seed for reproducibility -> same sequencies of random numbers


        PrintWriter out = new PrintWriter(System.out, true);

        for (int i = 0; i < K && generated + 2 * pointsPerCluster <= N; i++) {

            double cxA = 10 * i; // Center X for cluster i
            double cyA = 0;

            double cxB = 10 * i + 10;
            double cyB = 10;

            // Group A cluster
            for (int j = 0; j < pointsPerCluster; j++) {
                double x = cxA + rand.nextGaussian(); // Gaussian noise
                double y = cyA + rand.nextGaussian();
                out.printf("%.4f,%.4f,%s%n", x, y, "A");
                generated++;
            }

            // Group B far from group A in same cluster index
            for (int j = 0; j < pointsPerCluster; j++) {
                double x = cxB + rand.nextGaussian(); // Shifted center to the right
                double y = cyB + rand.nextGaussian(); // Shifted up
                out.printf("%.4f,%.4f,%s%n", x, y, "B");
                generated++;
            }

        }

        // Add some extra points if generated is less than N
        while (generated < N) {
            double x = rand.nextGaussian();
            double y = rand.nextGaussian();
            out.printf("%.4f,%.4f,%s%n", x, y, "A");
            generated++;
        }


        }


    }

