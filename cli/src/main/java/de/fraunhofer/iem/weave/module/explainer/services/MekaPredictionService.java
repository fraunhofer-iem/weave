package de.fraunhofer.iem.weave.module.explainer.services;

import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.List;

/**
 * PredictionService implementation that wraps a MEKA multi-label classifier.
 */
public class MekaPredictionService implements PredictionService {

    private final Instances header;
    private final MultiLabelClassifier classifier;
    private final int[] labelIndices;
    private final List<Integer> featureIndices;

    public MekaPredictionService(Instances header, MultiLabelClassifier classifier) throws Exception {
        this.header = new Instances(header, 0);
        this.classifier = classifier;

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
        int numLabels = labelIndices.length;
        double[][] out = new double[n][numLabels];

        for (int i = 0; i < n; i++) {
            double[] x = features[i];

            // Construct a new instance with the same attribute structure as the header.
            Instance inst = new DenseInstance(header.numAttributes());
            inst.setDataset(header);

            // Set feature values in the positions after the labels.
            for (int j = 0; j < featureIndices.size(); j++) {
                int attrIndex = featureIndices.get(j);
                inst.setValue(attrIndex, x[j]);
            }

            double[] probs = classifier.distributionForInstance(inst);
            out[i] = probs;
        }

        return out;
    }
}
