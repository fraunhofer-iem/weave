package de.fraunhofer.iem.weave.module.explainer.services;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.concurrent.CompletionException;
import java.util.stream.IntStream;

/**
 * PredictionService implementation that wraps a WEKA single-label Classifier.
 */
public class WekaPredictionService implements PredictionService {

    private final Instances header;
    private final Classifier classifier;
    private final int classIndex;

    /**
     * Constructs a new WekaPredictionService.
     * @param header     an Instances object used solely as a template;
     * @param classifier a trained WEKA Classifier that supports distributionForInstance().
     */
    public WekaPredictionService(Instances header, Classifier classifier) {
        // create a template for constructing new instances.
        this.header = new Instances(header, 0);
        this.classifier = classifier;
        this.classIndex = this.header.classIndex();
    }

    /**
     * Computes prediction probabilities for a batch of feature vectors using the underlying
     * WEKA Classifier.
     *
     * <p>Rows are scored in parallel. The classifier and the header template are only
     * read after training, and every row gets its own Instance, so no state is shared
     * between threads. SHAP drives this endpoint with batches of tens of thousands of
     * rows, and scoring them one at a time left all but one core idle.
     *
     * @param features a 2D array of shape [nSamples][nFeatures].
     * @return a 2D array of shape [nSamples][nOutputs], where each row contains the
     * predicted probabilities for that instance.
     */
    @Override
    public double[][] predictProba(double[][] features) throws Exception {
        int n = features.length;
        int numClasses = header.numClasses();
        double[][] out = new double[n][numClasses];

        try {
            IntStream.range(0, n).parallel().forEach(i -> {
                try {
                    // Distinct indices, so the concurrent writes do not race.
                    out[i] = classifier.distributionForInstance(toInstance(features[i]));
                } catch (Exception e) {
                    throw new CompletionException(e);
                }
            });
        } catch (CompletionException e) {
            if (e.getCause() instanceof Exception cause) {
                throw cause;
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