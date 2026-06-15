package de.fraunhofer.iem.weave.module.explainer.services;

import weka.classifiers.Classifier;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

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
     * @param features a 2D array of shape [nSamples][nFeatures].
     * @return a 2D array of shape [nSamples][nOutputs], where each row contains the
     * predicted probabilities for that instance.
     */
    @Override
    public synchronized double[][] predictProba(double[][] features) throws Exception {
        int n = features.length;
        int numClasses = header.numClasses();
        double[][] out = new double[n][numClasses];

        for (int i = 0; i < n; i++) {
            double[] x = features[i];

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
            double[] dist = classifier.distributionForInstance(inst);
            out[i] = dist;
        }
        return out;
    }
}