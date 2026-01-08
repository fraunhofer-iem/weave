package de.fraunhofer.weave.cli.module.explainer.shap.model;


import de.fraunhofer.weave.cli.PipelineOptions;
import de.fraunhofer.weave.cli.module.explainer.services.PredictionService;
import de.fraunhofer.weave.cli.module.explainer.services.WekaPredictionService;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class WekaModelForShap {

    private final Instances trainData;
    private final Classifier classifier;
    private final String[] featureNames;
    private final String positiveClassName;
    private final String testDatasetPath;

    public WekaModelForShap(PipelineOptions options, String classifierDescriptor) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(options.getDataset());
        this.testDatasetPath = options.getTestDatasetPath();
        Instances d = source.getDataSet();
        if (d.classIndex() < 0) {
            d.setClassIndex(d.numAttributes() - 1);
        }
        this.trainData = d;

        String[] opts = Utils.splitOptions(classifierDescriptor);
        this.classifier = (Classifier) Utils.forName(
                Classifier.class,
                opts[0],
                Arrays.copyOfRange(opts, 1, opts.length)
        );
        this.classifier.buildClassifier(this.trainData);

        int numAttrs = trainData.numAttributes();
        int classIdx = trainData.classIndex();
        List<String> fn = new ArrayList<>();
        for (int a = 0; a < numAttrs; a++) {
            if (a == classIdx) continue;
            fn.add(trainData.attribute(a).name());
        }
        this.featureNames = fn.toArray(new String[0]);

        if (trainData.numClasses() != 2) {
            throw new IllegalStateException(
                    "Expected binary classification (2 classes), found " + trainData.numClasses()
            );
        }
        this.positiveClassName = trainData.classAttribute().value(1);
    }

    /** Entire dataset as feature matrix (global explanations). */
    public double[][] getGlobalFeatureMatrix() {
        int n = trainData.numInstances();
        int numAttrs = trainData.numAttributes();
        int classIdx = trainData.classIndex();
        int numFeatures = numAttrs - 1;

        double[][] X = new double[n][numFeatures];

        for (int i = 0; i < n; i++) {
            Instance inst = trainData.instance(i);
            int f = 0;
            for (int a = 0; a < numAttrs; a++) {
                if (a == classIdx) continue;
                X[i][f++] = inst.value(a);
            }
        }
        return X;
    }

    /** Test file as feature matrix (local explanations only). */
    public double[][] getLocalFeatureMatrix() throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(this.testDatasetPath);
        Instances test = source.getDataSet();
        // assume same schema; class index as in training
        test.setClassIndex(trainData.classIndex());

        int n = test.numInstances();
        int numAttrs = test.numAttributes();
        int classIdx = test.classIndex();
        int numFeatures = numAttrs - 1;

        double[][] X = new double[n][numFeatures];

        for (int i = 0; i < n; i++) {
            Instance inst = test.instance(i);
            int f = 0;
            for (int a = 0; a < numAttrs; a++) {
                if (a == classIdx) continue;
                X[i][f++] = inst.value(a);
            }
        }
        return X;
    }

    public PredictionService toPredictionService() {
        return new WekaPredictionService(trainData, classifier);
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public String getPositiveClassName() {
        return positiveClassName;
    }
}
