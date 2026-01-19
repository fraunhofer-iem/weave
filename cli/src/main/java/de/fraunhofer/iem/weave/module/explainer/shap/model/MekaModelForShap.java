package de.fraunhofer.iem.weave.module.explainer.shap.model;


import de.fraunhofer.iem.weave.cli.PipelineOptions;
import de.fraunhofer.iem.weave.module.explainer.services.PredictionService;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.converters.ConverterUtils;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class MekaModelForShap {

    private static final Logger LOG = LoggerFactory.getLogger(MekaModelForShap.class);

    private final Instances trainData;
    private final MultiLabelClassifier classifier;
    private final int[] labelIndices;
    private final List<Integer> featureIndices;
    private final String[] featureNames;
    private final String[] labelNames;
    private final String testDatasetPath;


    public MekaModelForShap(PipelineOptions options, String classifierDescriptor) throws Exception {
        this.testDatasetPath = options.getTestDatasetPath();

        // 1) Load training data
        ConverterUtils.DataSource src = new ConverterUtils.DataSource(options.getDataset());
        Instances data = src.getDataSet();
        MLUtils.prepareData(data);
        this.trainData = data;
        LOG.info("[MEKA] Loaded training data: {} instances, {} attributes from {}",
                trainData.numInstances(), trainData.numAttributes(), options.getDataset());

        // 2) Determine label vs feature attributes
        int numLabels = inferNumLabels(trainData);
        if (numLabels <= 0) {
            throw new IllegalStateException("Could not determine number of labels for MEKA dataset.");
        }

        int numAttributes = trainData.numAttributes();
        this.labelIndices = new int[numLabels];
        this.featureIndices = new ArrayList<>();

        // Assumption: label attributes are the LAST numLabels attributes
        int labelPos = 0;
        for (int a = 0; a < numAttributes; a++) {
            if (a >= numAttributes - numLabels) {
                labelIndices[labelPos++] = a;
            } else {
                featureIndices.add(a);
            }
        }

        // build featureNames / labelNames
        this.featureNames = new String[featureIndices.size()];
        for (int i = 0; i < featureIndices.size(); i++) {
            featureNames[i] = trainData.attribute(featureIndices.get(i)).name();
        }

        this.labelNames = new String[labelIndices.length];
        for (int i = 0; i < labelIndices.length; i++) {
            labelNames[i] = trainData.attribute(labelIndices[i]).name();
        }

        LOG.info("[MEKA] Identified {} feature attributes and {} label attributes.",
                featureNames.length, labelNames.length);
        LOG.debug("[MEKA] Feature names: {}", Arrays.toString(featureNames));
        LOG.debug("[MEKA] Label names:   {}", Arrays.toString(labelNames));

        // 3) Reconstruct and train classifier
        String[] opts = Utils.splitOptions(classifierDescriptor);
        String classname = opts[0];
        String[] clsOpts = Arrays.copyOfRange(opts, 1, opts.length);

        MultiLabelClassifier cls = (MultiLabelClassifier) Utils.forName(
                MultiLabelClassifier.class, classname, clsOpts);
        cls.buildClassifier(trainData);

        this.classifier = cls;
        LOG.info("[MEKA] Reconstructed and trained classifier: {}", classifierDescriptor);
    }

    public double[][] getGlobalFeatureMatrix() {
        int n = trainData.numInstances();
        int d = featureIndices.size();
        double[][] X = new double[n][d];

        for (int i = 0; i < n; i++) {
            Instance inst = trainData.instance(i);
            for (int j = 0; j < d; j++) {
                int attIndex = featureIndices.get(j);
                X[i][j] = inst.value(attIndex);
            }
        }
        return X;
    }


    public double[][] getLocalFeatureMatrix() throws Exception {
        return getLocalFeatureMatrix(this.testDatasetPath);
    }


    public double[][] getLocalFeatureMatrix(String testDatasetPath) throws Exception {
        ConverterUtils.DataSource src = new ConverterUtils.DataSource(testDatasetPath);
        Instances test = src.getDataSet();
        MLUtils.prepareData(test);

        int n = test.numInstances();
        int d = featureIndices.size();
        double[][] X = new double[n][d];

        for (int i = 0; i < n; i++) {
            Instance inst = test.instance(i);
            for (int j = 0; j < d; j++) {
                int attIndex = featureIndices.get(j);
                X[i][j] = inst.value(attIndex);
            }
        }

        LOG.info("[MEKA] Built local feature matrix from {}: {} instances, {} features",
                testDatasetPath, n, d);
        return X;
    }

    public PredictionService toPredictionService() {
        return features -> {
            try {
                int numInstances = features.length;
                int numAttributes = trainData.numAttributes();

                // Create new Instances with same header as trainData
                Instances tmp = new Instances(trainData, numInstances);

                for (int i = 0; i < numInstances; i++) {
                    double[] vals = new double[numAttributes];
                    Arrays.fill(vals, Double.NaN); // labels remain missing

                    // Fill only the feature attributes from features[i][]
                    for (int j = 0; j < featureIndices.size(); j++) {
                        int attIndex = featureIndices.get(j);
                        vals[attIndex] = features[i][j];
                    }

                    Instance inst = new DenseInstance(1.0, vals);
                    tmp.add(inst);
                }

                // Call MEKA classifier for each instance
                double[][] probs = new double[numInstances][];
                for (int i = 0; i < numInstances; i++) {
                    probs[i] = classifier.distributionForInstance(tmp.instance(i));
                }

                return probs; // shape: (numInstances, numLabels)
            } catch (Exception e) {
                throw new RuntimeException("Error during MEKA prediction for SHAP", e);
            }
        };
    }

    private static int inferNumLabels(Instances data) {
        String relName = data.relationName();
        Pattern p = Pattern.compile("-C\\s*(\\d+)");
        Matcher m = p.matcher(relName);
        if (m.find()) {
            return Integer.parseInt(m.group(1));
        }
        throw new IllegalStateException(
                "Failed to infer number of labels from relation name: '" + relName +
                        "'. Expected to find a '-C <numLabels>' pattern.");
    }

    public String[] getFeatureNames() {
        return featureNames;
    }

    public String[] getLabelNames() {
        return labelNames;
    }

    public int[] getLabelIndices() {
        return labelIndices;
    }

    public List<Integer> getFeatureIndices() {
        return featureIndices;
    }

    public MultiLabelClassifier getClassifier() {
        return classifier;
    }
}
