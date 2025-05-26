import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.HashMap;
import java.util.Random;
import java.util.concurrent.Semaphore;

public class G01HW3 {

    class HashFunction{

        private int a, b, p = 8191, C;

        public HashFunction(int C, Random rnd){
            this.a = rnd.nextInt(p-1) + 1; // from 1 to p-1
            this.b = rnd.nextInt(p); // from 0 to p-1
            this.C = C;
        }

        public int hash(int x){
            return ((a * x + b) % p) % C;
        }

        public int sign(int x){
            return ((a * x + b) % p) % 2 == 0 ? 1 : -1;
        }
    }

    public static void main(String[] args) throws InterruptedException {

        // Lettura parametri da terminale
        if(args.length != 5){
            throw new IllegalArgumentException("USAGE: portExp T D W K");
        }

        // Configurazione Spark
        SparkConf conf = new SparkConf(true).setMaster("local[*]").setAppName("G01HW3");

        // Creazione del contesto di streaming
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        //Semaforo per gestire la terminazione
        Semaphore stopping = new Semaphore(1);
        stopping.acquire();

        // Lettura parametri
        int portExp = Integer.parseInt(args[0]);
        System.out.println("Receiving data from port = " + portExp);
        int T = Integer.parseInt(args[1]);
        System.out.println("Target number of items to be process = " + T);
        int D = Integer.parseInt(args[2]);
        System.out.println("Number of rows of each sketch = " + D);
        int W = Integer.parseInt(args[3]);
        System.out.println("Number of columns of each sketch = " + W);
        int K = Integer.parseInt(args[4]);
        System.out.println("Number of top frequent items of interest = " + K);

        // STATO GLOBALE DELL'ELABORAZIONE
        long[] streamLength = new long[1];
        streamLength[0] = 0L;
        HashMap<Long, Long> histogram = new HashMap<>();

        // Connection to the stream
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                .foreachRDD((batch, time) -> {});





        
    }
}
