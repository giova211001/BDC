import org.apache.hadoop.util.hash.Hash;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;

import java.util.*;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class DistinctItemsExample {

    // After how many items should we stop?
    // public static final int THRESHOLD = 1000000;

    /**
     *      SPIEGAZIONE DELLE PORTE
     *
     *      1) PORTE "NON DETERMINISTICHE" -> 8887 e 8889
     *          Porta 8887
     *          - Contenuto: pochi elementi molto frequenti, tutti gli altri elementi sono casuali
     *                       sull'intero dominio degli interi a 32 bit
     *          - Distribuzione: skewed (sbilanciata) -> molto rari, pochi ripetuti spesso
     *          - Uso tipico: simulare traffico reale con outlier frequenti (parole comuni in un testo..)
     *
     *          Porta 8889
     *          - Contenuto: pochi elementi molto frequenti, alcuni sono moderatamente frequenti, tutti gli
     *                       altri casuali
     *          - Distribuzione: meno sbilanciata rispetto a 8887
     *          - Uso tipico: testare algoritmi su dati più vari, dove ci sono gruppi con frequenze diverse
     *
     *      2) PORTE DETERMINISTICHE -> 8886 e 8888
     *      Queste porte generano sempre lo stesso stream, identico a ogni connessione. Utile per:
     *      - debug
     *      - ripetere test
     *      - confrontare algoritmi con condizioni identiche
     *
     *          Porta 8886
     *          - E' la versione deterministica della 8887
     *          - Pochi elementi molto frequenti + altri casuali, ma sempre gli stessi ogni volta che mi connetto
     *
     *          Porta 8888
     *          - E' la versione deterministica della 8889
     *          - Pochi elementi molto frequenti + alcuni mediamente frequenti + altri casuali ma strem è fisso
     *
     */

    public static void main(String[] args) throws Exception {

        /**
         *      MAIN E CONFIGURAZIONE INIZIALE
         *      Verifica che siano passati due argomenti da riga di comando:
         *      - port -> porta della socket da cui ricevere i dati
         *      - threshold -> massimo numero di elementi da elaborare
         */

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: port, threshold");
        }

        /**
         *      CREAZIONE DEL CONTESTO SPARK
         *      Crea la configurazione Spark
         *      - local[*] -> indica che il programma viene eseguito localmente usando tutti i
         *                    core disponibili
         */

        // IMPORTANT: the master must be set to "local[*]" or "local[n]" with n > 1, otherwise
        // there will be no processor running the streaming computation and your
        // code will crash with an out of memory (because the input keeps accumulating).
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]") // remove this line if running on the cluster
                .setAppName("DistinctExample");

        /**
         *      CREAZIONE DEL CONTESTO DI STREAMING
         *      Crea il contesto di streaming, cioè l'ambiente Spark che elabora le stream
         *      Ogni batch sarà di 100 millisecondi
         *      Imposta il livello di log per evitare un output troppo verboso
         *
         */

        // The definition of the streaming spark context  below, specifies the amount of
        // time used for collecting a batch, hence giving some control on the batch size.
        // Beware that the data generator we are using is very fast, so the suggestion is to
        // use batches of less than a second, otherwise you might exhaust the JVM memory.
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        /**
         *      CONTROLLO DELLA CONCORRENZA CON IL SEMAFORO
         *      Viene usato un semaforo per controllare la terminazione dell'app
         *      All'inizio viene bloccato (acquisito) il solo permesso disponibile
         *      Più avanti verrà rilasciato nel momento in cui si raggiunge la soglia (THRESHOLD) e sarà il
         *      segnale per chiudere l'elaborazione
         */

        // TECHNICAL DETAIL:
        // The streaming spark context and our code and the tasks that are spawned all
        // work concurrently. To ensure a clean shut down we use this semaphore. The 
        // main thread will first acquire the only permit available, and then it will try
        // to acquire another one right after spinning up the streaming computation.
        // The second attempt at acquiring the semaphore will make the main thread
        // wait on the call. Then, in the `foreachRDD` call, when the stopping condition
        // is met the semaphore is released, basically giving "green light" to the main
        // thread to shut down the computation. We cannot call `sc.stop()` directly in `foreachRDD`
        // because it might lead to deadlocks.

        // DETTAGLIO TECNICO:
        // Il contesto di streaming di Spark, il nostro codice e i task generati lavorano tutti
        // in parallelo. Per garantire una chiusura pulita utilizziamo questo semaforo.
        // Il thread principale inizialmente acquisisce l’unico permesso disponibile e poi
        // tenterà di acquisirne un altro subito dopo l’avvio del calcolo dello streaming.
        // Il secondo tentativo di acquisizione del semaforo farà sì che il thread principale
        // rimanga in attesa (bloccato).
        // Successivamente, nella chiamata `foreachRDD`, quando la condizione di arresto
        // viene soddisfatta, il semaforo viene rilasciato, dando di fatto il "via libera"
        // al thread principale per terminare l’elaborazione.
        // Non possiamo chiamare direttamente `sc.stop()` dentro `foreachRDD`
        // perché ciò potrebbe causare dei deadlock.


        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        int portExp = Integer.parseInt(args[0]);
        System.out.println("Receiving data from port = " + portExp);
        int THRESHOLD = Integer.parseInt(args[1]);
        System.out.println("Threshold = " + THRESHOLD);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        /**
         *      SETUP DELLE VARIABILI DI STATO DELLO STREAM
         *      - long streamlenght -> tiene il conteggio totale degli elementi processati. E' un array
         *                             per permettere la modifica di lambda, dove le variabili devono
         *                             essere finali
         *      - histogram -> mantiene tutte le chiavi distinte che sono apparse nello stream
         */

        // Variable streamLength below is used to maintain the number of processed stream items.
        // It must be defined as a 1-element array so that the value stored into the array can be
        // changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
        // long would not work since the lambda would not be allowed to update it.

        // La variabile streamLength qui sotto viene usata per mantenere il numero di elementi processati dello stream.
        // Deve essere definita come un array con un solo elemento affinché il valore memorizzato possa essere
        // modificato all'interno della lambda usata in foreachRDD. Usare un semplice contatore esterno di tipo long
        // non funzionerebbe perché la lambda non è autorizzata a modificarlo.
        long[] streamLength = new long[1]; // Stream length (an array to be passed by reference)
        streamLength[0]=0L;
        HashMap<Long, Long> histogram = new HashMap<>(); // Hash Table for the distinct elements


        /**
         *      STREAM DI DATI DALLA SOCKET
         *      - Riceve uno stream di testo da socket
         *      - @foreachRDD -> funzione eseguita per ogni batch
         *      - Ogni batch è un RDD contenente i dati ricevuti in quella finestra di 100 ms
         */

        // CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                // For each batch, to the following.
                // BEWARE: the `foreachRDD` method has "at least once semantics", meaning
                // that the same data might be processed multiple times in case of failure.
                .foreachRDD((batch, time) -> {
                    // this is working on the batch at time `time`.

                    /**
                     *      LOGICA DI ELABORAZIONE PER BATCH
                     *      Se non abbiamo ancora superato il limite THRESHOLD, allora:
                     *      - Calcola quanti elementi contiene il batch -> funzione count()
                     *      - Aggiorna il numero totale di elementi elaborati
                     *      
                     *
                     */
                    if (streamLength[0] < THRESHOLD) {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;
                        if (batchSize > 0) {
                            System.out.println("Batch size at time [" + time + "] is: " + batchSize);
                            // Extract the distinct items from the batch

                            /**
                             *      ELABORAZIONE DEL BATCH
                             *      - converte ogni riga (stringa) in Long
                             *      - Usa .reduceByKey((i1, i2) -> 1L) per rimuovere i duplicati nel batch (conserva solo chiavi uniche).
                             *      - raccoglie il risultato in una MapLocale
                             */
                            Map<Long, Long> batchItems = batch
                                    .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L))
                                    .reduceByKey((i1, i2) -> 1L) // rimuove i duplicati
                                    .collectAsMap();

                            /**
                             *      AGGIORNAMENTO DELLO STATO GLOBALE
                             *      - Aggiunge ogni nuovo valore distinto alla mappa histogram
                             */
                            
                            // Update the streaming state. If the overall count of processed items reaches the
                            // THRESHOLD value (among all batches processed so far), subsequent items of the
                            // current batch are ignored, and no further batches will be processed
                            for (Map.Entry<Long, Long> pair : batchItems.entrySet()) {
                                if (!histogram.containsKey(pair.getKey())) {
                                    histogram.put(pair.getKey(), 1L);
                                }
                            }

                            /**
                             *      CONTROLLO SOGLIA E STOP
                             *      Se hai raggiunto il limite, rilascia il semaforo -> segnala al thread principale di fermarsi
                             */
                            // If we wanted, here we could run some additional code on the global histogram
                            if (streamLength[0] >= THRESHOLD) {
                                // Stop receiving and processing further batches
                                stoppingSemaphore.release();
                            }

                        }
                    }
                });

        // MANAGING STREAMING SPARK CONTEXT
        System.out.println("Starting streaming engine");

        /**
         *      AVVIO E GESTIONE DELLA CHIUSURA
         *      
         */

        sc.start(); // avvia il processing
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire(); // attende che la soglia venga raggiunta
        System.out.println("Stopping the streaming engine");

        /* The following command stops the execution of the stream. The first boolean, if true, also
           stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
           the processing of all received data to be completed. You might get some error messages when
           the program ends, but they will not affect the correctness. You may also try to set the second
           parameter to true.
        */

        sc.stop(false, false); // ferma senza attendere le elaborazioni in corso
        System.out.println("Streaming engine stopped");

        /**
         *      STATISTICHE FINALI
         *      - stampa numero totale di elementi letti
         *      - stampa numero totale di elementi distinti
         */

        // COMPUTE AND PRINT FINAL STATISTICS
        System.out.println("Number of items processed = " + streamLength[0]);
        System.out.println("Number of distinct items = " + histogram.size());


        /**
         *      ELEMENTO PIU' GRANDE
         *      - ordina le chiavi in ordine decrescente
         *      - stampa il valore più grande tra quelli distinti ricevuti
         */
        long max = 0L;
        ArrayList<Long> distinctKeys = new ArrayList<>(histogram.keySet());
        Collections.sort(distinctKeys, Collections.reverseOrder());
        System.out.println("Largest item = " + distinctKeys.get(0));
    }
}
