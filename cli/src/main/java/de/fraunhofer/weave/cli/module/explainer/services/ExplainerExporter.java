package de.fraunhofer.weave.cli.module.explainer.services;


import de.fraunhofer.weave.cli.PipelineOptions;
import meka.core.MLUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.BufferedWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Utility class responsible for exporting feature matrices for SHAP-based
 * explainability.
 */
public class ExplainerExporter {

    private static final Logger logger = LoggerFactory.getLogger(ExplainerExporter.class);

    private final PipelineOptions options;

    /**
     * Creates a new ExplainerExporter instance.
     *
     */
    public ExplainerExporter(PipelineOptions options) {
        this.options = options;
    }

    /**
     * Exports the feature matrix of the full training dataset to a CSV file.
     *
     * @return the Path to the created global features CSV file.
     */
    public Path exportGlobalFeatures() throws Exception {
        String datasetPath = options.getDataset();
        if (datasetPath == null || datasetPath.isBlank()) {
            throw new IllegalStateException("Training dataset path is not configured in ExrmConfigOptions.");
        }

        String toolkit = options.getToolkit();
        Path out = Paths.get("global_features.csv");

        if ("weka".equalsIgnoreCase(toolkit)) {
            exportWekaFeatures(datasetPath, out);
        } else if ("meka".equalsIgnoreCase(toolkit)) {
            exportMekaFeatures(datasetPath, out);
        } else {
            throw new IllegalArgumentException("Unsupported toolkit for global feature export: " + toolkit);
        }

        logger.info("Exported global features to {}", out.toAbsolutePath());
        return out;
    }

    /**
     * Exports the feature matrix of the test dataset to a CSV file.
     * @return the Path to the created local features CSV file.
     */
    public Path exportLocalFeatures() throws Exception {
        String testDatasetPath = options.getTestDatasetPath();
        if (testDatasetPath == null || testDatasetPath.isBlank()) {
            throw new IllegalStateException("Test dataset path is not configured in ExrmConfigOptions.");
        }

        String toolkit = options.getToolkit();
        Path out = Paths.get("local_features.csv");

        if ("weka".equalsIgnoreCase(toolkit)) {
            exportWekaFeatures(testDatasetPath, out);
        } else if ("meka".equalsIgnoreCase(toolkit)) {
            exportMekaFeatures(testDatasetPath, out);
        } else {
            throw new IllegalArgumentException("Unsupported toolkit for local feature export: " + toolkit);
        }

        logger.info("Exported local features to {}", out.toAbsolutePath());
        return out;
    }

    /**
     * Loads a WEKA single-label dataset from the given path, writes the features to the specified
     * CSV file (only non-class attributes).
     *
     * @param datasetPath the path to the WEKA dataset (e.g., ARFF file).
     * @param outputPath  the path where the CSV file should be written.
     */
    private void exportWekaFeatures(String datasetPath, Path outputPath) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
        Instances data = source.getDataSet();

        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        int numAttrs = data.numAttributes();
        int classIdx = data.classIndex();

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            // Write header: names of all non-class attributes.
            int featureCount = 0;
            for (int attrIndex = 0; attrIndex < numAttrs; attrIndex++) {
                if (attrIndex == classIdx) {
                    continue;
                }
                if (featureCount++ > 0) {
                    writer.write(",");
                }
                writer.write(data.attribute(attrIndex).name());
            }
            writer.newLine();

            // Write rows: values of all non-class attributes for each instance.
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                featureCount = 0;
                for (int attrIndex = 0; attrIndex < numAttrs; attrIndex++) {
                    if (attrIndex == classIdx) {
                        continue;
                    }
                    if (featureCount++ > 0) {
                        writer.write(",");
                    }
                    writer.write(Double.toString(inst.value(attrIndex)));
                }
                writer.newLine();
            }
        }
    }

    /**
     * Loads a MEKA multi-label dataset from the given path and writes
     * all features to the specified CSV file (only multi-label data).
     *
     * @param datasetPath the path to the MEKA dataset (e.g., ARFF file).
     * @param outputPath  the path where the CSV file should be written.
     */
    private void exportMekaFeatures(String datasetPath, Path outputPath) throws Exception {
        ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
        Instances data = source.getDataSet();

        // Prepare as multi-label: after this, classIndex() = number of labels L,
        // and labels are at indices 0..L-1; features follow from L..numAttributes-1.
        MLUtils.prepareData(data);
        int L = data.classIndex();
        int numFeatures = data.numAttributes() - L;

        try (BufferedWriter writer = Files.newBufferedWriter(outputPath, StandardCharsets.UTF_8)) {
            // Write header: names of feature attributes from L..end.
            for (int j = 0; j < numFeatures; j++) {
                int attrIndex = L + j;
                if (j > 0) {
                    writer.write(",");
                }
                writer.write(data.attribute(attrIndex).name());
            }
            writer.newLine();

            // Write rows: values of feature attributes for each instance.
            for (int i = 0; i < data.numInstances(); i++) {
                Instance inst = data.instance(i);
                for (int j = 0; j < numFeatures; j++) {
                    int attrIndex = L + j;
                    if (j > 0) {
                        writer.write(",");
                    }
                    writer.write(Double.toString(inst.value(attrIndex)));
                }
                writer.newLine();
            }
        }
    }
}
