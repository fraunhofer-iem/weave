package de.fraunhofer.weave.cli;

import picocli.CommandLine;

import java.util.concurrent.Callable;

@CommandLine.Command(name = "weave", mixinStandardHelpOptions = true,
        version = "weave-1.0", description = "")
public class CliOptions implements Callable<Integer> {

    @CommandLine.Option(names = {"-t", "--toolkit"}, description = {"ML toolkit: weka, meka, scikit"})
    private String mlToolkit = "weka";

    @CommandLine.Option(names = {"-c", "--config"}, description = {"Experiment configuration file"})
    private String configFile = "config.properties";

    @CommandLine.Option(names = {"-X", "--explainer-only"}, description = {"Run only the explainer using the model provided via --explain. "})
    private boolean explainerOnly = false;

    @CommandLine.Option(names = {"-e", "--explain"}, description = {"Custom model to explain"})
    private String  explainModel = "";

    @Override
    public Integer call() throws Exception {
        return new PipelineRunner().run(new PipelineOptions(mlToolkit, configFile, explainModel, explainerOnly));
    }
}
