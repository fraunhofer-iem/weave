package de.fraunhofer.iem.weave.cli;

import de.fraunhofer.iem.weave.module.ModelExplainer;
import de.fraunhofer.iem.weave.module.ModelSelector;
import de.fraunhofer.iem.weave.module.ModelExperimenter;
import de.fraunhofer.iem.weave.module.explainer.services.MekaPredictionService;
import de.fraunhofer.iem.weave.module.explainer.services.PredictionService;
import de.fraunhofer.iem.weave.module.explainer.services.WekaPredictionService;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.util.Arrays;
import java.util.concurrent.CancellationException;

public class PipelineRunner {

    private static final Logger logger = LoggerFactory.getLogger(PipelineRunner.class);
    private PipelineOptions exrmOptions;

    public int run(PipelineOptions options) {

        String classifierDescriptor = null;

        try {
            boolean explainerOnly = options.isExplainerOnly();

            if (explainerOnly) {
                // Mode 2: run only the explainer
                classifierDescriptor = options.getExplainModel();
                if (classifierDescriptor == null || classifierDescriptor.isBlank()) {
                    throw new IllegalArgumentException(
                            "Explainer-only mode requires a non-empty model descriptor via --explain."
                    );
                }
                logger.info("Explainer-only mode. Using model: {}", classifierDescriptor);
            } else {
                // Mode 1: run selector, experimenter and explainer
                logger.info("Running full pipeline.");

                //Model selection with ML(2)-Plan
                ModelSelector modelSelector = new ModelSelector(options);
                modelSelector.run();
                classifierDescriptor = modelSelector.getClassifier();
                logger.info("Selected model: {}", classifierDescriptor);

                //Empirical evaluation with jaicaore-experiments
                ModelExperimenter modelExperimenter =
                        new ModelExperimenter(options, classifierDescriptor);
                modelExperimenter.evaluate();
            }

            String toolkit = options.getToolkit();
            PredictionService predictor;

            if ("weka".equalsIgnoreCase(toolkit)) {
                predictor = buildWekaPredictionService(options, classifierDescriptor);
            } else if ("meka".equalsIgnoreCase(toolkit)) {
                predictor = buildMekaPredictionService(options, classifierDescriptor);
            } else {
                logger.error("Toolkit '{}' is not supported.", toolkit);
                return 1;
            }

            ModelExplainer explainer = new ModelExplainer(predictor, options);
            explainer.runHttpExplainer();
            explainer.runShapExplainer(options.getPythonPath(),options.getPythonExplainerPath(), options.getOutputPath());

            return 0;
        } catch (
                CancellationException e) {
            logger.warn("Analysis run was cancelled");
            return 66;
        } catch (
                Exception e) {
            logger.error("Analysis run terminated with error", e);
            return 500;
        }
    }

    /**
     * Builds a PredictionService for a WEKA (single-label) classifier based on
     * the provided dataset path and the classifier.
     */
    private PredictionService buildWekaPredictionService(PipelineOptions options,
                                                         String classifierDescriptor) throws Exception {
        String datasetPath = options.getDataset();
        if (datasetPath == null || datasetPath.isBlank()) {
            throw new IllegalStateException("Training dataset path is not configured.");
        }

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
        Instances data = source.getDataSet();
        if (data.classIndex() < 0) {
            data.setClassIndex(data.numAttributes() - 1);
        }

        String[] opts = Utils.splitOptions(classifierDescriptor);
        Classifier wekaClassifier = (Classifier) Utils.forName(
                Classifier.class,
                opts[0],
                Arrays.copyOfRange(opts, 1, opts.length)
        );
        wekaClassifier.buildClassifier(data);

        logger.info("Built WEKA classifier for explanation: {}", classifierDescriptor);
        return new WekaPredictionService(data, wekaClassifier);
    }

    /**
     * Builds a PredictionService for a MEKA (multi-label) classifier based on
     * the dataset path and the classifier descriptor string.
     */
    private PredictionService buildMekaPredictionService(PipelineOptions options,
                                                         String classifierDescriptor) throws Exception {
        String datasetPath = options.getDataset();
        if (datasetPath == null || datasetPath.isBlank()) {
            throw new IllegalStateException("Training dataset path is not configured in ExrmConfigOptions.");
        }

        ConverterUtils.DataSource source = new ConverterUtils.DataSource(datasetPath);
        Instances data = source.getDataSet();
        MLUtils.prepareData(data);

        String[] opts = Utils.splitOptions(classifierDescriptor);
        MultiLabelClassifier mekaClassifier = (MultiLabelClassifier) Utils.forName(
                MultiLabelClassifier.class,
                opts[0],
                Arrays.copyOfRange(opts, 1, opts.length)
        );
        mekaClassifier.buildClassifier(data);

        logger.info("Built MEKA classifier for explanation: {}", classifierDescriptor);
        return new MekaPredictionService(data, mekaClassifier);
    }
}
