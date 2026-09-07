package de.fraunhofer.iem.weave.module.explainer.services;

import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.AbstractClassifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * PredictionService implementation that wraps a MEKA multi-label classifier.
 */
public class MekaPredictionService implements PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(MekaPredictionService.class);

    /** Worker count, and therefore the number of classifier copies held in memory. */
    private static final int THREADS = Integer.getInteger("weave.predict.threads",
            Runtime.getRuntime().availableProcessors());

    private final Instances header;
    private final MultiLabelClassifier prototype;
    private final int[] labelIndices;
    private final List<Integer> featureIndices;

    /**
     * One classifier per worker thread. MEKA transformation methods wrap a WEKA base
     * learner, and several WEKA classifiers mutate internal Filter state during
     * prediction, so a shared instance is not safe to call concurrently. Copying per
     * thread avoids having to reason about which combination happens to be safe.
     */
    private final ThreadLocal<MultiLabelClassifier> perThread;

    /** Dedicated pool so the copies live on a fixed, known set of threads. */
    private final ForkJoinPool pool;

    /** False when the classifier cannot be copied; then scoring is serial and locked. */
    private final boolean parallel;

    public MekaPredictionService(Instances header, MultiLabelClassifier classifier) throws Exception {
        this.header = new Instances(header, 0);
        this.prototype = classifier;

        MLUtils.prepareData(this.header);
        int L = this.header.classIndex(); // number of labels

        // Label attributes 0..L-1
        this.labelIndices = new int[L];
        for (int i = 0; i < L; i++) {
            this.labelIndices[i] = i;
        }

        // Feature attributes
        this.featureIndices = new ArrayList<>();
        for (int attrIndex = L; attrIndex < this.header.numAttributes(); attrIndex++) {
            this.featureIndices.add(attrIndex);
        }

        boolean copyable;
        try {
            copy(classifier);
            copyable = true;
        } catch (Exception e) {
            copyable = false;
            logger.warn("{} cannot be copied ({}); scoring will run single-threaded.",
                    classifier.getClass().getName(), e.getMessage());
        }
        this.parallel = copyable;
        this.pool = copyable ? new ForkJoinPool(THREADS) : null;
        this.perThread = ThreadLocal.withInitial(() -> {
            try {
                return copy(prototype);
            } catch (Exception e) {
                throw new IllegalStateException("Could not copy classifier for this thread", e);
            }
        });
        if (copyable) {
            logger.info("Scoring with up to {} threads, one classifier copy each"
                    + " (override with -Dweave.predict.threads=N).", THREADS);
        }
    }

    private static MultiLabelClassifier copy(MultiLabelClassifier c) throws Exception {
        return (MultiLabelClassifier) AbstractClassifier.makeCopy(c);
    }

    /**
     * Computes prediction probabilities for a batch of feature vectors using the
     * MEKA MultiLabelClassifier.
     *
     * @param features a 2D array of shape [nSamples][nFeatures].
     * @return a 2D array of shape [nSamples][nLabels], where each row contains the prediction
     * probabilities P(label_j = 1) for j from 0 to nLabels-1.
     */
    @Override
    public double[][] predictProba(double[][] features) throws Exception {
        int n = features.length;
        double[][] out = new double[n][labelIndices.length];

        if (!parallel) {
            synchronized (prototype) {
                for (int i = 0; i < n; i++) {
                    out[i] = prototype.distributionForInstance(toInstance(features[i]));
                }
            }
            return out;
        }

        try {
            // Distinct indices, so the concurrent writes do not race.
            pool.submit(() -> IntStream.range(0, n).parallel().forEach(i -> {
                try {
                    out[i] = perThread.get().distributionForInstance(toInstance(features[i]));
                } catch (Exception e) {
                    throw new CompletionException(e);
                }
            })).get();
        } catch (ExecutionException e) {
            Throwable cause = e.getCause();
            if (cause instanceof CompletionException && cause.getCause() != null) {
                cause = cause.getCause();
            }
            if (cause instanceof Exception ex) {
                throw ex;
            }
            throw e;
        }
        return out;
    }

    /**
     * Builds an Instance for a single feature vector. Labels occupy indices 0..L-1 and
     * are left unset; features follow from L onwards. The header is shared but only
     * read, so this is safe to call concurrently.
     */
    private Instance toInstance(double[] x) {
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);

        for (int j = 0; j < featureIndices.size(); j++) {
            inst.setValue(featureIndices.get(j), x[j]);
        }
        return inst;
    }
}
