package de.fraunhofer.weave.cli.module.explainer.services;

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
}
