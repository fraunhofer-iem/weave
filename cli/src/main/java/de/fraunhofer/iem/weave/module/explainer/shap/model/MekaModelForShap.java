package de.fraunhofer.iem.weave.module.explainer.shap.model;


import de.fraunhofer.iem.weave.cli.PipelineOptions;
import de.fraunhofer.iem.weave.module.explainer.services.MekaPredictionService;
import de.fraunhofer.iem.weave.module.explainer.services.PredictionService;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/***
 * Prepare the MEKA model for explaining with Shap.
 */

public class MekaModelForShap {

    private final Instances trainData;
    private final MultiLabelClassifier classifier;
    private final int[] labelIndices;
    private final List<Integer> featureIndices;
    private final String[] featureNames;
    private final String[] labelNames;
    private final String testDatasetPath;

    public MekaModelForShap(PipelineOptions options, String classifierDescriptor) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(options.getDataset());
        Instances d = source.getDataSet();
        this.testDatasetPath = options.getTestDatasetPath();

        // Prepare dataset and assign labels
        MLUtils.prepareData(d);
        this.trainData = d;
        int L = trainData.classIndex();
        this.labelIndices = new int[L];
        for (int i = 0; i < L; i++) {
            this.labelIndices[i] = i;
        }

        this.featureIndices = new ArrayList<>();
        List<String> fn = new ArrayList<>();
        List<String> ln = new ArrayList<>();

        for (int a = 0; a < trainData.numAttributes(); a++) {
            if (a < L) {
                // label
                ln.add(trainData.attribute(a).name());
            } else {
                // feature
                featureIndices.add(a);
                fn.add(trainData.attribute(a).name());
            }
        }

        this.featureNames = fn.toArray(new String[0]);
        this.labelNames = ln.toArray(new String[0]);

        String[] opts = Utils.splitOptions(classifierDescriptor);
        this.classifier = (MultiLabelClassifier) Utils.forName(
                MultiLabelClassifier.class,
                opts[0],
                Arrays.copyOfRange(opts, 1, opts.length)
        );
        this.classifier.buildClassifier(this.trainData);
    }

    //Global: entire dataset feature matrix.
    public double[][] getGlobalFeatureMatrix() {
        int n = trainData.numInstances();
        int numFeatures = featureIndices.size();
        double[][] X = new double[n][numFeatures];

        for (int i = 0; i < n; i++) {
            Instance inst = trainData.instance(i);
            for (int j = 0; j < numFeatures; j++) {
                int a = featureIndices.get(j);
                X[i][j] = inst.value(a);
            }
        }
        return X;
    }

    //Local: test file as feature matrix.
    public double[][] getLocalFeatureMatrix() throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(this.testDatasetPath);
        Instances test = source.getDataSet();
        MLUtils.prepareData(test);

        int n = test.numInstances();
        int numFeatures = featureIndices.size();
        double[][] X = new double[n][numFeatures];

        for (int i = 0; i < n; i++) {
            Instance inst = test.instance(i);
            for (int j = 0; j < numFeatures; j++) {
                int a = featureIndices.get(j);
                X[i][j] = inst.value(a);
            }
        }
        return X;
    }

    public PredictionService toPredictionService() throws Exception {
        return new MekaPredictionService(trainData, classifier);
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public String[] getLabelNames() {
        return labelNames;
    }
}
