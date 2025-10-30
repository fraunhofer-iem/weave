package de.fraunhofer.exrm;

import de.fraunhofer.exrm.cli.ExrmConfigOptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CancellationException;

public class RunTool {

    private static final Logger logger = LoggerFactory.getLogger(RunTool.class);
    private ExrmConfigOptions exrmOptions;


    public int run(ExrmConfigOptions options) {

        try {

            //Model selection with ML(2)-Plan
            ModelSelector modelSelector = new ModelSelector(options);
            modelSelector.run();

            //Empirical evaluation with jaicaore-experiments
            ModelExperimenter modelExperimenter = new ModelExperimenter(options, modelSelector.getClassifier());
            modelExperimenter.evaluate();


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
