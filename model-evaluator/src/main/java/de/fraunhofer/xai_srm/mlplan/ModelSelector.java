package de.fraunhofer.xai_srm.mlplan;

import ai.libs.jaicore.ml.classification.loss.dataset.EClassificationPerformanceMeasure;
import ai.libs.jaicore.ml.classification.multilabel.dataset.IMekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.dataset.MekaInstances;
import ai.libs.jaicore.ml.classification.multilabel.evaluation.loss.InstanceWiseF1;
import ai.libs.jaicore.ml.classification.multilabel.learner.IMekaClassifier;
import ai.libs.jaicore.ml.core.evaluation.evaluator.SupervisedLearnerExecutor;
import ai.libs.jaicore.ml.core.filter.SplitterUtil;
import ai.libs.jaicore.ml.scikitwrapper.IScikitLearnWrapper;
import ai.libs.jaicore.ml.weka.classification.learner.IWekaClassifier;
import ai.libs.jaicore.ml.weka.dataset.IWekaInstances;
import ai.libs.jaicore.ml.weka.dataset.WekaInstances;
import ai.libs.mlplan.core.MLPlan;
import ai.libs.mlplan.meka.ML2PlanMekaBuilder;
import ai.libs.mlplan.sklearn.builder.MLPlanScikitLearnBuilder;
import ai.libs.mlplan.weka.MLPlanWekaBuilder;
import meka.core.MLUtils;
import org.api4.java.ai.ml.classification.multilabel.evaluation.IMultiLabelClassification;
import org.api4.java.ai.ml.classification.singlelabel.evaluation.ISingleLabelClassification;
import org.api4.java.ai.ml.core.dataset.supervised.ILabeledDataset;
import org.api4.java.ai.ml.core.evaluation.execution.ILearnerRunReport;
import org.api4.java.algorithm.Timeout;
import weka.core.Instances;

import java.io.FileReader;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Selects machine learning model using ML-Plan/ML2-Plan.
 *
 * @author Oshando Johnson on 27.09.20
 */
public class ModelSelector {

    private static final Logger LOGGER = LoggerFactory.getLogger(ModelSelector.class.getName());

    public static void main(String[] args) throws Exception {

        LOGGER.info("Args: TOOLKIT {}, CPU {}, Dataset {}, Duration {}, Seed {}", args[0], args[1], args[2], args[3], args[4]);

        switch (args[0]) {
            case "weka":
                Weka weka = new Weka();
                weka.selectModel(Integer.parseInt(args[1]), args[2], Long.parseLong(args[3]), Long.parseLong(args[4]));
            case "meka":
                Meka meka = new Meka();
                meka.selectModel(Integer.parseInt(args[1]), args[2], Long.parseLong(args[3]), Long.parseLong(args[4]));
            case "scikit":
                SciKit sciKit = new SciKit();
                sciKit.selectScikitLearnModel(Integer.parseInt(args[1]), args[2], Long.parseLong(args[3]), Long.parseLong(args[4]));
        }
    }






}
