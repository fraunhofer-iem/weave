package de.fraunhofer.weave.cli.module;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.*;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import de.fraunhofer.weave.cli.PipelineOptions;
import meka.classifiers.multilabel.Evaluation;
import meka.classifiers.multilabel.MultiLabelClassifier;
import meka.core.MLUtils;
import meka.core.Result;
import org.aeonbits.owner.ConfigFactory;
import org.api4.java.algorithm.exceptions.AlgorithmExecutionCanceledException;
import org.api4.java.algorithm.exceptions.AlgorithmTimeoutedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.net.URL;
import java.util.*;

public class ModelExperimenter {

    private static final Logger logger = LoggerFactory.getLogger(ModelExperimenter.class);

    private PipelineOptions options;
    private IDatabaseConfig dbConfig;
    private IExperimentSetConfig expConfig;
    private String classifier;

    public ModelExperimenter(PipelineOptions options, String classifier) {

        this.options = options;
        this.classifier = classifier;
        expConfig = (IExperimentSetConfig) ConfigFactory
                .create(IExperimentSetConfig.class)
                .loadPropertiesFromFile(new File(options.getExperimentConfigFile()));

        dbConfig = (IDatabaseConfig) ConfigFactory
                .create(IDatabaseConfig.class)
                .loadPropertiesFromFile(new File(options.getDbConfigFile()));
    }

    public void evaluate() {

        ExperimenterMySQLHandle handle = new ExperimenterMySQLHandle(dbConfig);

        ExperimentDatabasePreparer preparer = new ExperimentDatabasePreparer(expConfig, handle);
        try {
            preparer.synchronizeExperiments();
        } catch (ExperimentDBInteractionFailedException | IllegalExperimentSetupException |
                 AlgorithmTimeoutedException | InterruptedException | AlgorithmExecutionCanceledException |
                 ExperimentAlreadyExistsInDatabaseException e) {
            logger.error("Couldn't synchronize experiment table.", e);
        }

        //Get evaluator for WEKA or MEKA
        IExperimentSetEvaluator evaluator = switch (options.getToolkit()) {
            case "weka" -> getWekaEvaluator(options.getDataset(), classifier);
            case "meka" -> getMekaEvaluator(options.getDataset(), classifier);
            case "scikit" -> getSciKitEvaluator(options.getDataset(), classifier);
            default -> null;
        };

        try {
            ExperimentRunner runner = new ExperimentRunner(expConfig, evaluator, handle);
            runner.sequentiallyConductExperiments(-1);
        } catch (ExperimentDBInteractionFailedException | InterruptedException e) {
            logger.error("Error trying to run experiments.", e);
            System.exit(1);
        }
    }

    public IExperimentSetEvaluator getMekaEvaluator(String dataset, String classifierDescriptor) {

        IExperimentSetEvaluator evaluator =
                (ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) -> {
                    Experiment experiment = experimentEntry.getExperiment();
                    Map<String, String> keyFields = experiment.getValuesOfKeyFields();

                    try {

                        int seed = Integer.parseInt(keyFields.get("seeds"));

                        // Load dataset
                        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataset);
                        Instances data = source.getDataSet();
                        data.randomize(new Random(seed));
                        MLUtils.prepareData(data);

                        if (keyFields.get("classifier").contentEquals("automl")) {
                            keyFields.replace("classifier", classifierDescriptor);
                        }

                        //create the classifier
                        String classifier = keyFields.get("classifier");
                        String[] options = Utils.splitOptions(classifier);
                        MultiLabelClassifier mekaClassifier = (MultiLabelClassifier) Utils.forName(MultiLabelClassifier.class, options[0], Arrays.copyOfRange(options, 1, options.length));

                        Result result = Evaluation.cvModel(mekaClassifier, data, 10, "PCutL", "7");

                        String info = result.info.toString();

                        double[] lblPrecision = (double[]) result.getMeasurement("Precision (per label)");
                        double[] lblRecall = (double[]) result.getMeasurement("Recall (per label)");
                        double[] lblHarmonic = (double[]) result.getMeasurement("Harmonic (per label)");

                        //Macro calculates for each class individually and then takes the average of these values.
                        double avgPrecision = (double) result.getMeasurement("Macro Precision");
                        double avgRecall = (double) result.getMeasurement("Macro Recall");
                        double avgMicroF1Measure = (double) result.getMeasurement("F1 (micro averaged)");
                        double avgMacroF1MeasureLbl = (double) result.getMeasurement("F1 (macro averaged by label)");
                        double avgMacroF1MeasureEx = (double) result.getMeasurement("F1 (macro averaged by example)");

                        //It calculates the precision globally by counting the total true positives and false positives.
                        double microPrecision = (double) result.getMeasurement("Micro Precision");
                        double microRecall = (double) result.getMeasurement("Micro Recall");

                        // submit the results:
                        Map<String, Object> experimentResults = new HashMap<>();
                        experimentResults.put("automlClassifier", classifier);
                        experimentResults.put("info", info);

                        experimentResults.put("lblPrecision", Arrays.toString(lblPrecision));
                        experimentResults.put("lblRecall", Arrays.toString(lblRecall));
                        experimentResults.put("lblHarmonic", Arrays.toString(lblHarmonic));

                        experimentResults.put("avgPrecision", avgPrecision);
                        experimentResults.put("avgRecall", avgRecall);

                        experimentResults.put("avgMicroF1Measure", avgMicroF1Measure);
                        experimentResults.put("avgMacroF1MeasureLbl", avgMacroF1MeasureLbl);
                        experimentResults.put("avgMacroF1MeasureEx", avgMacroF1MeasureEx);

                        experimentResults.put("microPrecision", microPrecision);
                        experimentResults.put("microRecall", microRecall);

                        processor.processResults(experimentResults);

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
        return evaluator;
    }


    /**
     * Evaluates SciKit-learn ML model on the provided dataset.
     *
     * @param dataset
     * @param classifierDescriptor
     * @return
     */
    public IExperimentSetEvaluator getSciKitEvaluator(String dataset, String classifierDescriptor) {

        IExperimentSetEvaluator evaluator =
                (ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) -> {
                    Experiment experiment = experimentEntry.getExperiment();
                    Map<String, String> keyFields = experiment.getValuesOfKeyFields();

                    try {

                        // gather experiment key values:
                        int seed = Integer.parseInt(keyFields.get("seeds"));

                        if (keyFields.get("classifier").contentEquals("automl")) {
                            keyFields.replace("classifier", classifierDescriptor);
                        }

                        String classifier = keyFields.get("classifier");

                        URL url = ModelExperimenter.class.getClassLoader()
                                .getResource("scripts/experimenter/scikit-experimenter.py");
                        File f = new File(Objects.requireNonNull(url).toURI());

                        ProcessBuilder processBuilder = new ProcessBuilder("python",
                                f.getAbsolutePath(),
                                "--arff", options.getDataset(),
                                "--model", classifier,
                                "--seed", Integer.toString(seed));
                        processBuilder.redirectErrorStream(true);

                        System.out.println(processBuilder.command());

                        Process process = processBuilder.start();

                        StringBuilder sb = new StringBuilder();
                        for (int ch; (ch = process.getInputStream().read()) != -1; ) {
                            sb.append((char) ch);
                        }

                        System.out.println(sb.toString());

                        int exitCode = process.waitFor();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
        return evaluator;
    }

    public IExperimentSetEvaluator getWekaEvaluator(String dataset, String classifierDescriptor) {

        IExperimentSetEvaluator evaluator =
                (ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) -> {
                    Experiment experiment = experimentEntry.getExperiment();
                    Map<String, String> keyFields = experiment.getValuesOfKeyFields();

                    // gather experiment key values:
                    int seed = Integer.parseInt(keyFields.get("seeds"));

                    if (keyFields.get("classifier").contentEquals("automl")) {
                        keyFields.replace("classifier", classifierDescriptor);
                    }

                    String classifier = keyFields.get("classifier");

                    try {
                        // Load dataset
                        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataset);

                        Instances data = source.getDataSet();
                        data.setClassIndex(data.numAttributes() - 1);
                        data.randomize(new Random(seed));

                        String approach = dataset.substring(dataset.lastIndexOf("/") + 1, dataset.indexOf("."));

                        //create the classifier
                        String[] options = Utils.splitOptions(classifier);
                        Classifier wekaClassifier = (Classifier) Utils.forName(Classifier.class, options[0], Arrays.copyOfRange(options, 1, options.length));

                        weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(data);
                        eval.crossValidateModel(wekaClassifier, data, 10, new Random(seed));

                        String info = approach + " " + eval.toClassDetailsString();

                        int numClasses = data.numClasses();

                        double[] lblTruePositive = new double[numClasses];
                        double[] lblFalsePositive = new double[numClasses];
                        double[] lblTrueNegative = new double[numClasses];
                        double[] lblFalseNegative = new double[numClasses];
                        double[] lblPrecision = new double[numClasses];
                        double[] lblRecall = new double[numClasses];
                        double[] lblHarmonic = new double[numClasses];

                        for (int i = 0; i < numClasses; i++) {

                            lblTruePositive[i] = eval.numTruePositives(i);
                            lblFalsePositive[i] = eval.numFalsePositives(i);
                            lblTrueNegative[i] = eval.numTrueNegatives(i);
                            lblFalseNegative[i] = eval.numFalseNegatives(i);
                            lblPrecision[i] = (Double.isNaN(eval.precision(i)) ? 0.0 : eval.precision(i));
                            lblRecall[i] = (Double.isNaN(eval.recall(i)) ? 0.0 : eval.recall(i));

                            lblHarmonic[i] = (Double.isNaN(eval.fMeasure(i)) ? 0.0 : eval.fMeasure(i));
                        }

                        //Macro calculates for each class individually and then takes the average of these values.
                        double avgPrecision = Arrays.stream(lblPrecision).average().getAsDouble();
                        double avgRecall = Arrays.stream(lblRecall).average().getAsDouble();
                        double avgFMeasure = (Double.isNaN(eval.unweightedMacroFmeasure()) ? 0.0 : eval.unweightedMacroFmeasure());

                        //It calculates the precision globally by counting the total true positives and false positives.
                        double microRecall = Arrays.stream(lblTruePositive).sum() / (Arrays.stream(lblTruePositive).sum() + Arrays.stream(lblFalseNegative).sum());
                        double microPrecision = Arrays.stream(lblTruePositive).sum() / (Arrays.stream(lblTruePositive).sum() + Arrays.stream(lblFalsePositive).sum());
                        double microFMeasure = eval.unweightedMicroFmeasure();
                        double weightedFMeasure = (Double.isNaN(eval.weightedFMeasure()) ? 0.0 : eval.weightedFMeasure());

                        // submit the results:
                        Map<String, Object> experimentResults = new HashMap<>();
                        experimentResults.put("automlClassifier", classifier);
                        experimentResults.put("info", info);
                        experimentResults.put("lblPrecision", Arrays.toString(lblPrecision));
                        experimentResults.put("lblRecall", Arrays.toString(lblRecall));
                        experimentResults.put("lblHarmonic", Arrays.toString(lblHarmonic));
                        experimentResults.put("lblTruePositive", Arrays.toString(lblTruePositive));
                        experimentResults.put("lblFalsePositive", Arrays.toString(lblFalsePositive));
                        experimentResults.put("lblTrueNegative", Arrays.toString(lblTrueNegative));
                        experimentResults.put("lblFalseNegative", Arrays.toString(lblFalseNegative));

                        experimentResults.put("avgPrecision", avgPrecision);
                        experimentResults.put("avgRecall", avgRecall);
                        experimentResults.put("avgFMeasure", avgFMeasure);

                        experimentResults.put("microRecall", microRecall);
                        experimentResults.put("microPrecision", microPrecision);
                        experimentResults.put("microFMeasure", microFMeasure);
                        experimentResults.put("weightedFMeasure", weightedFMeasure);

                        logger.info("Evaluation results: {}", experimentResults);
                        processor.processResults(experimentResults);

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
        return evaluator;
    }
}