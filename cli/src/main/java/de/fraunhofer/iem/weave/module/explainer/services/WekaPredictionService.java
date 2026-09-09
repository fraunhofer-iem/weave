package de.fraunhofer.iem.weave.module.explainer.services;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Attribute;
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
 * PredictionService implementation that wraps a WEKA single-label Classifier.
 */
public class WekaPredictionService implements PredictionService {

    private static final Logger logger = LoggerFactory.getLogger(WekaPredictionService.class);

    /** Worker count, and therefore the number of classifier copies held in memory. */
    private static final int THREADS = Integer.getInteger("weave.predict.threads",
            Runtime.getRuntime().availableProcessors());

    private final Instances header;
    private final Classifier prototype;
    private final int classIndex;

    /**
     * One classifier per worker thread. Several WEKA classifiers are NOT safe to call
     * concurrently: SMO, SimpleLogistic and Logistic push each instance through an
     * internal Filter whose Queue is shared mutable state, so sharing a single
     * instance across threads throws "Queue is empty" or silently returns the wrong
     * distribution. Tree ensembles happen to be safe, but that is not something the
     * Classifier interface promises, so every thread gets its own deep copy.
     */
    private final ThreadLocal<Classifier> perThread;

    /** Dedicated pool so the copies live on a fixed, known set of threads. */
    private final ForkJoinPool pool;

    /** False when the classifier cannot be copied; then scoring is serial and locked. */
    private final boolean parallel;

    /**
     * Constructs a new WekaPredictionService.
     * @param header     an Instances object used solely as a template;
     * @param classifier a trained WEKA Classifier that supports distributionForInstance().
     */
    public WekaPredictionService(Instances header, Classifier classifier) {
        // create a template for constructing new instances.
        this.header = new Instances(header, 0);
        this.prototype = classifier;
        this.classIndex = this.header.classIndex();

        boolean copyable;
        try {
            AbstractClassifier.makeCopy(classifier);
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
                return AbstractClassifier.makeCopy(prototype);
            } catch (Exception e) {
                throw new IllegalStateException("Could not copy classifier for this thread", e);
            }
        });
        if (copyable) {
            logger.info("Scoring with up to {} threads, one classifier copy each"
                    + " (override with -Dweave.predict.threads=N).", THREADS);
        }
    }

    /**
     * The class values in the order distributionForInstance returns them, which is the
     * order the ARFF class attribute declares them in. A three-class problem such as
     * sscm (tag = None/Target/Input) is explained per class, and these are the names
     * that identify which column is which.
     */
    @Override
    public List<String> getOutputNames() {
        Attribute classAttribute = header.classAttribute();
        if (!classAttribute.isNominal()) {
            return List.of();
        }
        List<String> names = new ArrayList<>(classAttribute.numValues());
        for (int i = 0; i < classAttribute.numValues(); i++) {
            names.add(classAttribute.value(i));
        }
        return names;
    }

    /**
     * Computes prediction probabilities for a batch of feature vectors using the underlying
     * WEKA Classifier.
     *
     * @param features a 2D array of shape [nSamples][nFeatures].
     * @return a 2D array of shape [nSamples][nOutputs], where each row contains the
     * predicted probabilities for that instance.
     */
    @Override
    public double[][] predictProba(double[][] features) throws Exception {
        int n = features.length;
        double[][] out = new double[n][header.numClasses()];

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
     * Builds a WEKA Instance for a single feature vector, skipping the class attribute.
     * The header is shared but only read, so this is safe to call concurrently.
     */
    private Instance toInstance(double[] x) {
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);

        int featureIndex = 0;
        for (int attrIndex = 0; attrIndex < header.numAttributes(); attrIndex++) {
            if (attrIndex == classIndex) {
                // Skip the classIndex at prediction time.
                continue;
            }
            inst.setValue(attrIndex, x[featureIndex++]);
        }
        return inst;
    }
}
