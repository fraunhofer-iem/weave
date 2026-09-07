package de.fraunhofer.iem.weave.module.explainer.services;

import java.util.List;

/**
 * A generic prediction service interface that provides a probability prediction method for
 * batches of feature vectors.
 */
public interface PredictionService {

    /**
     * Computes prediction probabilities for a batch of feature vectors.
     * @param features a 2D array of shape [nSamples][nFeatures].
     * @return a 2D array of shape [nSamples][nOutputs], where each row contains the
     * predicted probabilities for that instance.
     */
    double[][] predictProba(double[][] features) throws Exception;

    /**
     * Names for the columns predictProba returns, in the same order. Lets the SHAP
     * explainer label its per-class output with the class values from the ARFF
     * instead of bare indices. Empty when the implementation has no names to give,
     * in which case the explainer falls back to indices.
     */
    default List<String> getOutputNames() {
        return List.of();
    }
}
