package de.fraunhofer.xai_srm.experiment;

import ai.libs.jaicore.db.IDatabaseConfig;
import ai.libs.jaicore.experiments.*;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterMySQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentAlreadyExistsInDatabaseException;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
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
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.meta.RandomSubSpace;
import weka.classifiers.trees.LMT;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.util.*;

public class RepeatedCVExperimenter {

    private static final Logger logger = LoggerFactory.getLogger(RepeatedCVExperimenter.class);

    public static void main(final String[] args) {

        IExperimentSetConfig expConfig = (IExperimentSetConfig) ConfigFactory
                .create(IExperimentSetConfig.class)

                .loadPropertiesFromFile(new File(args[0] + "/experiments.cnf"));

        IDatabaseConfig dbConfig = (IDatabaseConfig) ConfigFactory
                .create(IDatabaseConfig.class)
                .loadPropertiesFromFile(new File(args[0] + "/db.properties"));

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
        IExperimentSetEvaluator evaluator = switch (args[1]) {
            case "weka" -> getWekaEvaluator();
            case "meka" -> getMekaEvaluator();
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

    public static IExperimentSetEvaluator getMekaEvaluator() {

        IExperimentSetEvaluator evaluator =
                (ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) -> {
                    Experiment experiment = experimentEntry.getExperiment();
                    Map<String, String> keyFields = experiment.getValuesOfKeyFields();

                    // gather experiment key values:
                    String dataset = keyFields.get("datasets");
                    String classifierName = keyFields.get("classifiers");
                    String options = keyFields.get("options");
                    int seed = Integer.parseInt(keyFields.get("seeds"));

                    try {
                        // Load dataset
                        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataset);
                        Instances data = source.getDataSet();
                        data.randomize(new Random(seed));
                        MLUtils.prepareData(data);

                        //create the classifier
                        MultiLabelClassifier classifier = (MultiLabelClassifier) Class.forName(classifierName)
                                .getDeclaredConstructor().newInstance();
                        classifier.setOptions(options.split("\\s+"));

                        Result result = Evaluation.cvModel(classifier, data, 10, "PCutL", "7");

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

    public static IExperimentSetEvaluator getWekaEvaluator() {

        IExperimentSetEvaluator evaluator =
                (ExperimentDBEntry experimentEntry, IExperimentIntermediateResultProcessor processor) -> {
                    Experiment experiment = experimentEntry.getExperiment();
                    Map<String, String> keyFields = experiment.getValuesOfKeyFields();

                    // gather experiment key values:
                    String dataset = keyFields.get("datasets");
                    int seed = Integer.parseInt(keyFields.get("seeds"));

                    try {
                        // Load dataset
                        ConverterUtils.DataSource source = new ConverterUtils.DataSource(dataset);
                        Instances data = source.getDataSet();
                        data.setClassIndex(data.numAttributes() - 1);

                        data.randomize(new Random(seed));

                        String approach = dataset.substring(dataset.lastIndexOf("/") + 1, dataset.indexOf("."));

                        weka.classifiers.Evaluation eval = new weka.classifiers.Evaluation(data);
                        eval.crossValidateModel(getClassifier(approach), data, 10, new Random(seed));

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
                            ;
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
                        System.out.println(experimentResults);
                        processor.processResults(experimentResults);

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                };
        return evaluator;
    }

    public static Classifier getClassifier(String classifierName) throws Exception {

        switch (classifierName) {
            case "source":
                ClassificationViaRegression source = new ClassificationViaRegression();
                source.setOptions(new String[]{"-W", "weka.classifiers.meta.RandomCommittee", "-do-not-check-capabilities"
                        , "--", "-S", "1", "-num-slots", "1", "-I", "24", "-W", "weka.classifiers.trees.RandomForest",
                        "-do-not-check-capabilities", "--", "-P", "100", "-I", "100", "-num-slots", "1",
                        "-do-not-check-capabilities", "-K", "0", "-M", "8.0", "-V", "0.001", "-S", "1"});
                return source;
            case "sink":
                RandomSubSpace sink = new RandomSubSpace();
                sink.setOptions(new String[]{"-P", "0.53125", "-S", "1", "-num-slots", "1", "-I", "39", "-W",
                        "weka.classifiers.lazy.IBk", "-do-not-check-capabilities",
                        "--", "-K", "1", "-W", "0", "-A", "weka.core.neighboursearch.LinearNNSearch"});
                return sink;
            case "sanitizer":
                ClassificationViaRegression sanitizer = new ClassificationViaRegression();
                sanitizer.setOptions(new String[]{"-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "56", "-W",
                        "weka.classifiers.lazy.IBk", "-do-not-check-capabilities", "--", "-K", "7",
                        "-W", "0", "-X", "-A", "weka.core.neighboursearch.LinearNNSearch", "-do-not-check-capabilities"});
                return sanitizer;
            case "cwe79":
                Bagging cwe79 = new Bagging();
                cwe79.setOptions(new String[]{"-P", "100", "-O", "-S", "1", "-num-slots", "1", "-I", "28", "-W",
                        "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "2.280153483153162",
                        "-M", "34", "-do-not-check-capabilities", "-num-decimal-places", "4"});
                return cwe79;
            case "cwe89":
                LWL cwe89 = new LWL();
                cwe89.setOptions(new String[]{"-U", "0", "-K", "60",
                        "-W", "weka.classifiers.meta.RandomCommittee", "-do-not-check-capabilities",
                        "--", "-S", "1", "-num-slots", "1", "-I", "10", "-W",
                        "weka.classifiers.trees.RandomTree",
                        "-do-not-check-capabilities", "--", "-K", "0", "-M", "3.0", "-V", "100.0", "-S", "1", "-depth",
                        "10", "-U", "-do-not-check-capabilities"});
                return cwe89;

            case "authentication":
                Bagging auth = new Bagging();
                auth.setOptions(new String[]{"-P", "179", "-S", "1", "-num-slots", "1", "-I",
                        "67", "-W", "weka.classifiers.trees.LMT", "-do-not-check-capabilities",
                        "--", "-R", "-P", "-I", "-1", "-M", "60", "-W", "0.5625",
                        "-do-not-check-capabilities"});
                return auth;

            case "cwe863":
                Bagging cwe863 = new Bagging();
                cwe863.setOptions(new String[]{"-P", "174", "-S", "1", "-num-slots", "1", "-I",
                        "95", "-W", "weka.classifiers.functions.Logistic", "-do-not-check-capabilities", "--", "-R", "0.681699067355134", "-M", "23", "-do-not-check-capabilities", "-num-decimal-places", "4"});
                return cwe863;

            case "cwe862":
                Logistic cwe862 = new Logistic();
                cwe862.setOptions(new String[]{"-R", "0.040003508330387474", "-M", "5",
                        "-do-not-check-capabilities", "-num-decimal-places", "4"});
                return cwe862;


            case "cwe306":
                LMT cwe306 = new LMT();
                cwe306.setOptions(new String[]{"-C", "-P", "-I", "-1", "-M", "42", "-W",
                        "0.27083333333333337", "-A", "-do-not-check-capabilities"});
                return cwe306;


            case "cwe78":
                Bagging cwe78 = new Bagging();
                cwe78.setOptions(new String[]{"-P", "79", "-S", "1", "-num-slots", "1", "-I", "102",
                        "-W", "weka.classifiers.trees.RandomTree", "-do-not-check-capabilities", "--",
                        "-K", "0", "-M", "36.0", "-V", "10.0", "-S", "1", "-do-not-check-capabilities"});
                return cwe78;


            case "cwe601":
                RandomSubSpace cwe601 = new RandomSubSpace();
                cwe601.setOptions(new String[]{"-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "10",
                        "-W", "weka.classifiers.rules.PART", "-do-not-check-capabilities", "--",
                        "-M", "2", "-C", "0.25", "-Q", "1", "-do-not-check-capabilities"});
                return cwe601;


            case "sscm":
                RandomSubSpace sscm = new RandomSubSpace();
                sscm.setOptions(new String[]{"-P", "0.5", "-S", "1", "-num-slots", "1", "-I", "56", "-W",
                        "weka.classifiers.lazy.IBk", "-do-not-check-capabilities", "--", "-K", "7", "-W", "0", "-X", "-A",
                        "weka.core.neighboursearch.LinearNNSearch",
                        "-do-not-check-capabilities"});
                return sscm;
            default:
                SMO smo = new SMO();
                return smo;
        }
    }
}