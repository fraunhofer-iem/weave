package de.fraunhofer.xai_srm.mlplan;

import ai.libs.jaicore.ml.classification.multilabel.dataset.IMekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.dataset.MekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.evaluation.loss.InstanceWiseF1;
import ai.libs.jaicore.ml.classification.multilabel.learner.IMekaClassifier;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.meka.ML2PlanMekaBuilder;
import meka.core.MLUtils;
import org.api4.java.ai.ml.classification.multilabel.evaluation.IMultiLabelClassification;
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

public class  Meka {

    private final Logger LOGGER = LoggerFactory.getLogger(Meka.class.getName());

    /**
     * Run ML2-Plan to select multi-label model.
     *
     * @param arffFile path to ARFF dataset file
     * @throws Exception
     */
    public void selectModel(int cpu, String arffFile, long duration, long seed) throws Exception {

        Instances instances = new Instances(new FileReader(arffFile));
        LOGGER.info("Loaded {} instances from {}", instances.numInstances(), arffFile);
        //Prepare instances and split into train and test datasets
        MLUtils.prepareData(instances);

        IMekaInstances dataset = new MekaInstances(instances);
        List<ILabeledDataset<?>> split = SplitterUtil.getSimpleTrainTestSplit(dataset, new Random(seed), .7);
        LOGGER.info("Loading {} instances from {}", instances.numInstances(), arffFile);

        // Initialize ML2-Plan
        MLPlan<IMekaClassifier> mlplan = new ML2PlanMekaBuilder()
                .withNumCpus(cpu)
                .withTimeOut(new Timeout(duration, TimeUnit.MINUTES))
                .withDataset(split.get(0)).build();

        try {

            //Evaluate ML2-Plan solution with test set
            IMekaClassifier classifier = mlplan.call();

            SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();
            ILearnerRunReport report = executor.execute(classifier, split.get(1));
            LOGGER.info("Model error Rate {}", new InstanceWiseF1()
                    .loss(report.getPredictionDiffList().getCastedView(int[].class, IMultiLabelClassification.class)));

        } catch (NoSuchElementException e) {

            LOGGER.error("Building the classifier failed: {}", e.getMessage());
        }
    }
}
