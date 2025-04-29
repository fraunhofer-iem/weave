package de.fraunhofer.xai_srm.mlplan;

import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import meka.core.MLUtils;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.algorithm.Timeout;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Instances;

import java.io.FileReader;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

public class Weka {

    private final Logger LOGGER = LoggerFactory.getLogger(Weka.class.getName());

    /**
     * Run ML-Plan to select multi-label model.
     *
     * @param arffFile path to ARFF dataset file
     * @throws Exception
     */
    public void selectModel(int cpu, String arffFile, long duration, long seed) throws Exception {

        Instances instances = new Instances(new FileReader(arffFile));
        LOGGER.info("Loaded {} instances from {}", instances.numInstances(), arffFile);
        //Prepare instances and split into train and test datasets

        MLUtils.prepareData(instances);
        IWekaInstances dataset = new WekaInstances(instances);
        List<ILabeledDataset<?>> split = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, new Random(seed), .7);
        LOGGER.info("Loading {} instances from {}", instances.numInstances(), arffFile);

        // Initialize ML2-Plan
        MLPlan<IWekaClassifier> mlplan = new MLPlanWekaBuilder()
                .withNumCpus(cpu)
                .withTimeOut(new Timeout(duration, TimeUnit.MINUTES))
                .withDataset(split.get(0)).build();

        try {

            //Evaluate ML-Plan solution with test set
            IWekaClassifier classifier = mlplan.call();

            LOGGER.info("Chosen model is: {}", (mlplan.getSelectedClassifier()));
            LOGGER.info("Chosen WEKA model is: {}", (classifier.getClassifier()));

            /* evaluate solution produced by mlplan */
            SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
            ILearnerRunReport report = executor.execute(classifier, split.get(1));
            LOGGER.info("Error Rate of the solution produced by ML-Plan: {}. Internally believed error was {}",
                    EClassificationPerformanceMeasure.ERRORRATE.loss(report.getPredictionDiffList()
                            .getCastedView(Integer.class, ISingleLabelClassification.class)),
                    mlplan.getInternalValidationErrorOfSelectedClassifier());
        } catch (NoSuchElementException e) {

            LOGGER.error("Building the classifier failed: {}", e.getMessage());
        }
    }
}
