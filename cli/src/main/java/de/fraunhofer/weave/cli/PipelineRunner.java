package de.fraunhofer.weave.cli;

import de.fraunhofer.weave.cli.module.ModelExplainer;
import de.fraunhofer.weave.cli.module.ModelSelector;
import de.fraunhofer.weave.cli.module.ModelExperimenter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CancellationException;

public class PipelineRunner {

    private static final Logger logger = LoggerFactory.getLogger(PipelineRunner.class);
    private PipelineOptions exrmOptions;

    public int run(PipelineOptions options) {

        try {

            //Model selection with ML(2)-Plan
            ModelSelector modelSelector = new ModelSelector(options);
            modelSelector.run();

            //Empirical evaluation with jaicaore-experiments
            ModelExperimenter modelExperimenter = new ModelExperimenter(options, modelSelector.getClassifier());
            modelExperimenter.evaluate();

            ModelExplainer modelExplainer = new ModelExplainer(options, modelSelector.getClassifier());
            modelExplainer.explain();

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
}
