package de.fraunhofer.iem.weave.module;

import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.classification.multilabel.dataset.IMekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.dataset.MekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.learner.IMekaClassifier;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.scikitwrapper.IScikitLearnWrapper;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.meka.ML2PlanMekaBuilder;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import de.fraunhofer.iem.weave.cli.PipelineOptions;
import meka.core.MLUtils;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.algorithm.Timeout;
import weka.core.Instances;

import java.io.FileReader;
import java.util.List;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Utils;

/**
 * Selects machine learning model using ML-Plan/ML2-Plan.
 *
 * @author Oshando Johnson on 27.09.20
 */
public class ModelSelector {

    private static final Logger LOGGER = LoggerFactory.getLogger(ModelSelector.class.getName());

    private PipelineOptions configOptions;
    private List<ILabeledDataset<?>> labeledDataset;
    private String classifier;
    private ILearnerRunReport runReport;

    public ModelSelector(PipelineOptions options) {

        configOptions = options;
        LOGGER.info("ML-Plan Options: {}", configOptions);

        try {
            //Prepare instances and split into train and test datasets
            Instances instances = new Instances(new FileReader(configOptions.getDataset()));
            LOGGER.info("Loaded {} instances from {}", instances.numInstances(), configOptions.getDataset());

            MLUtils.prepareData(instances);

            switch (configOptions.getToolkit()) {
                case "weka":
                case "scikit":
                    labeledDataset = SplitterUtil.getLabelStratifiedTrainTestSplit(new WekaInstances(instances), configOptions.getSeed(), .7);
                    break;
                case "meka":
                    IMekaInstances dataset = new MekaInstances(instances);
                    //labeledDataset = SplitterUtil.getLabelStratifiedTrainTestSplit(dataset, new Random(configOptions.getSeed()), .7);
                    labeledDataset = SplitterUtil.getSimpleTrainTestSplit(dataset, configOptions.getSeed(), 0.7);
                    break;
            }
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void run() throws Exception {

        SupervisedLearnerExecutor executor = new SupervisedLearnerExecutor();

        switch (configOptions.getToolkit()) {

            case "weka":
                // Initialize ML-Plan
                LOGGER.info("Starting search phase");
                MLPlan<IWekaClassifier> mlplan = new MLPlanWekaBuilder()
                        .withNumCpus(configOptions.getCpu())
                        .withSeed(configOptions.getSeed())
                        .withNodeEvaluationTimeOut(new Timeout(configOptions.getNodeTimeout(), TimeUnit.MINUTES))
                        .withCandidateEvaluationTimeOut(new Timeout(configOptions.getCandidateTimeout(), TimeUnit.MINUTES))
                        .withTimeOut(new Timeout(configOptions.getTimeout(), TimeUnit.MINUTES))
                        .withDataset(labeledDataset.get(0)).build();

                IWekaClassifier wekaClassifier = mlplan.call();
                LOGGER.info("ML-Plan Selected Classifier: {}", (mlplan.getSelectedClassifier()));
                LOGGER.info("WEKA Classifier: {}", (wekaClassifier.getClassifier()));

                classifier = Utils.toCommandLine(wekaClassifier.getClassifier());
                LOGGER.info("Command line WEKA Pipeline: {}", classifier);

                runReport = executor.execute(wekaClassifier, labeledDataset.get(1));
                LOGGER.info("Error Rate of the solution produced by ML-Plan: {}. Internally believed error was {}",
                        EClassificationPerformanceMeasure.ERRORRATE.loss(runReport.getPredictionDiffList().getCastedView(Integer.class, ISingleLabelClassification.class)), mlplan.getInternalValidationErrorOfSelectedClassifier());
                break;

            case "meka":

                // Initialize ML2-Plan
                MLPlan<IMekaClassifier> ml2plan = new ML2PlanMekaBuilder()
                        .withNumCpus(configOptions.getCpu())
                        .withSeed(configOptions.getSeed())
                        .withNodeEvaluationTimeOut(new Timeout(configOptions.getNodeTimeout(), TimeUnit.MINUTES))
                        .withCandidateEvaluationTimeOut(new Timeout(configOptions.getCandidateTimeout(), TimeUnit.MINUTES))
                        .withTimeOut(new Timeout(configOptions.getTimeout(), TimeUnit.MINUTES))
                        .withDataset(labeledDataset.get(0)).build();

                IMekaClassifier mekaClassifier = ml2plan.call();
                LOGGER.info("ML2-Plan Selected Classifier: {}", (ml2plan.getSelectedClassifier()));
                LOGGER.info("MEKA Classifier: {}", (mekaClassifier.getClassifier()));

                classifier = Utils.toCommandLine(mekaClassifier.getClassifier());
                LOGGER.info("Command line MEKA Pipeline: {}", classifier);

                //Evaluate ML2-Plan solution with test set
                runReport = executor.execute(mekaClassifier, labeledDataset.get(1));
                break;

            case "scikit":
                // Initialize ML-Plan
                MLPlan<IScikitLearnWrapper> scikit = MLPlanScikitLearnBuilder.forClassification()
                        .withNumCpus(configOptions.getCpu())
                        .withNodeEvaluationTimeOut(new Timeout(configOptions.getNodeTimeout(), TimeUnit.MINUTES))
                        .withCandidateEvaluationTimeOut(new Timeout(configOptions.getCandidateTimeout(), TimeUnit.MINUTES))
                        .withTimeOut(new Timeout(configOptions.getTimeout(), TimeUnit.MINUTES))
                        .withDataset(labeledDataset.get(0)).build();

                //Evaluate ML-Plan solution with test set
                IScikitLearnWrapper scikitClassifier = scikit.call();
                LOGGER.info("SciKit model selected: {}", scikitClassifier);
                classifier = scikitClassifier.toString();

                //Evaluate ML2-Plan solution with test set
                runReport = executor.execute(scikitClassifier, labeledDataset.get(1));
                break;
        }
    }

    public ILearnerRunReport getRunReport() {
        return runReport;
    }

    public String getClassifier() {
        return classifier;
    }

    public void setClassifier(String classifier) {
        this.classifier = classifier;
    }
}
