import cats.kernel.Hash;
import org.apache.hadoop.fs.shell.Count;
import org.apache.hadoop.yarn.webapp.example.MyApp;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Duration;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Int;
import scala.Tuple2;

import java.util.*;

import java.util.concurrent.Semaphore;
import java.util.stream.Collectors;

public class G01HW3 {

    public static class Countminsketch {
        public int[][] CM;
        public final int D, W;
        public final HashFunction[] hf;

        public Countminsketch(int depth, int width, HashFunction[] h) {
            D = depth;
            W = width;
            hf = h;
            CM = new int[D][W];
        }

        public void add(int x)
        {
            int cindex;
            for (int i = 0; i < D; i++) {
                cindex = hf[i].hash(x);
                CM[i][cindex]++;
            }
        }
        public void add_f(int x, int f)
        {
            int cindex;
            for (int i = 0; i < D; i++)
            {
                cindex = hf[i].hash(x);
                CM[i][cindex] += f;
            }
        }

        public int estimate(int x) {
            int cindex = hf[0].hash(x);
            int freq = CM[0][cindex];
            for (int i = 1; i < D; i++) {
                cindex = hf[i].hash(x);
                if (CM[i][cindex] < freq) freq = CM[i][cindex];
            }
            return freq;
        }

        public void merge(Countminsketch cms) {
            if (D != cms.D || W != cms.W) throw new IllegalArgumentException("CMS dimension mismatch");
            for (int i = 0; i < D; i++) {
                if (!hf[i].compare(cms.hf[i])) throw new IllegalArgumentException("CMS hash mismatch");
            }

            for (int i = 0; i < D; i++) {
                for (int j = 0; j < W; j++) CM[i][j] += cms.CM[i][j];
            }

        }
    }


        public static class Countsketch
        {
            public int[][] CS;
            public final int D, W;
            public final HashFunction[] hf;
            public final HashFunction[] shf;

            public Countsketch(int depth, int width, HashFunction[] h, HashFunction[] sh)
            {
                D = depth;
                W = width;
                hf = h;
                shf = sh;
                CS = new int[D][W];
            }
            public void add(int x)
            {
                int cindex, sgn;
                for(int i = 0; i < D; i++)
                {
                    cindex = hf[i].hash(x);
                    sgn = shf[i].sign(x);

                    CS[i][cindex] += sgn;
                }
            }

            public void add_f(int x, int f)
            {
                int cindex, sgn;
                for(int i = 0; i < D; i++)
                {
                    cindex = hf[i].hash(x);
                    sgn = shf[i].sign(x);

                    CS[i][cindex] += sgn * f;
                }
            }

            public int estimate(int x)
            {
                int[] estimates = new int[D];
                int cindex, sign;
                for (int i = 0; i < D; i++) {
                    cindex = hf[i].hash(x);
                    sign = shf[i].sign(x);
                    estimates[i] = CS[i][cindex] * sign;
                }
                Arrays.sort(estimates);
                if(D % 2 == 0) return (estimates[D/2 - 1] + estimates[D/2])/2;
                return estimates[D/2]; // mediana
            }

            public void merge(Countsketch cms)
            {
                if(D !=  cms.D || W != cms.W) throw new IllegalArgumentException("CMS dimension mismatch");
                for(int i = 0; i < D; i++)
                {
                    if(!hf[i].compare(cms.hf[i]) || (!shf[i].compare(cms.shf[i]))) throw new IllegalArgumentException("CMS hash mismatch");
                }

                for (int i = 0; i < D; i++)
                {
                    for (int j = 0; j < W; j++) CS[i][j] += cms.CS[i][j];
                }
            }
    }
    public static class HashFunction
    {
        public int a, b, p = 8191, C;
        public HashFunction(int C, Random rnd){
            this.a = rnd.nextInt(p-1) + 1; // from 1 to p-1
            this.b = rnd.nextInt(p); // from 0 to p-1
            this.C = C;
        }
        public int hash(int x){return Math.floorMod((a * x + b) % p, C);}
        public int sign(int x){return ((a * x + b) % p) % 2 == 0 ? 1 : -1;}
        public boolean compare(HashFunction h) {return (a == h.a && b == h.b && C == h.C);}
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



        // STRUTTURE DATI PER CM E CS + HASH FUNCTION



        HashFunction[] CM_hash = new HashFunction[D];
        HashFunction[] CS_hash = new HashFunction[D];
        HashFunction[] CS_hash_sgn = new HashFunction[D];

        Random r = new Random();
        for(int i = 0; i < D; i++){
            CM_hash[i] = new HashFunction(W, r);
            CS_hash[i] = new HashFunction(W, r);
            CS_hash_sgn[i] = new HashFunction(W, r);
        }

        Countminsketch cms = new Countminsketch(D, W, CM_hash);
        Countsketch cs = new Countsketch(D, W, CS_hash, CS_hash_sgn);
        // STATO GLOBALE DELL'ELABORAZIONE
        long[] streamLength = new long[1];
        streamLength[0] = 0L;


        Map<Integer, Integer> mymap = new HashMap<>();
        Map<Integer, Integer> histogram = new HashMap<>();
        // Connection to the stream
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                .foreachRDD((batch, time) -> {
                    if(streamLength[0] < T)
                    {
                        long batch_size = batch.count();
                        streamLength[0] += batch_size;
                        Map<Integer, Integer> items = batch.mapToPair(s -> new Tuple2<>(Integer.parseInt(s), 1))
                                .reduceByKey((i1, i2) -> i1 + i2)
                                .collectAsMap();

                        int element, freq;
                        for (Map.Entry<Integer, Integer> entry : items.entrySet())
                        {
                            element = entry.getKey();
                            freq = entry.getValue();
                            cms.add_f(element, freq);
                            cs.add_f(element, freq);
                            mymap.compute(entry.getKey(), (k, v) -> v == null ? entry.getValue() : v + entry.getValue());
                            if (!histogram.containsKey(element)) histogram.put(element, 1);
                        }
                        if (streamLength[0] >= T) stopping.release();
                    }
                });


        sc.start();
        stopping.acquire();

        sc.stop(false, false);
        sc.awaitTermination();

        System.out.println("Number of processed items = " + streamLength[0]);
        System.out.println("Number of distinct items = " + histogram.size());
        List<Map.Entry<Integer, Integer>> top_k = mymap.entrySet()
                .stream()
                .sorted((e1, e2) -> e2.getValue().compareTo(e1.getValue())).limit(K)
                .collect(Collectors.toList());

        System.out.println("Number of Top-K Heavy Hitters = " + top_k.size());

        int  item;
        double true_freq, est_freq_CM, est_freq_CS, rel_err_CM, rel_err_CS, avg_rel_err_CM = 0, avg_rel_err_CS = 0;
        for (Map.Entry<Integer, Integer> entry : top_k)
        {
            item = entry.getKey();
            true_freq = entry.getValue();
            est_freq_CM = cms.estimate(item);
            est_freq_CS = cs.estimate(item);
            rel_err_CM = (est_freq_CM - true_freq) / true_freq;
            rel_err_CS = Math.abs(est_freq_CS - true_freq) / true_freq;
            avg_rel_err_CM += rel_err_CM;
            avg_rel_err_CS += rel_err_CS;
        }
        avg_rel_err_CM = avg_rel_err_CM / top_k.size();
        avg_rel_err_CS = avg_rel_err_CS/ top_k.size();
        System.out.println("Avg Relative Error for TOP-K Heavy Hitters with CM = " + avg_rel_err_CM);
        System.out.println("Avg Relative Error for TOP-K Heavy Hitters with CS = " + avg_rel_err_CS);

        if(K <= 10)
        {
            top_k.sort(Comparator.comparing(Map.Entry::getKey));
            for (Map.Entry<Integer, Integer> entry : top_k)
            {
                item = entry.getKey();
                true_freq = entry.getValue();
                est_freq_CM = cms.estimate(entry.getKey());

                System.out.println("Item " + item + " True Frequency = " + (int)true_freq + " Estimated Frequency with CM = " + (int)est_freq_CM);
            }
        }



    }
}





