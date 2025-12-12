package de.fraunhofer.weave.cli;

import picocli.CommandLine;

import java.util.concurrent.Callable;

@CommandLine.Command(name = "weave", mixinStandardHelpOptions = true,
        version = "weave-1.0", description = "")
public class CliOptions implements Callable<Integer> {

    @CommandLine.Option(names = {"-t", "--toolkit"}, description = {"ML toolkit: weka, meka, scikit"})
    private String mlToolkit = "meka";

    @CommandLine.Option(names = {"-c", "--config"}, description = {"Experiment configuration file"})
    private String configFile = "config.file";

    @Override
    public Integer call() throws Exception {
        return new PipelineRunner().run(new PipelineOptions(mlToolkit, configFile));
    }
}
