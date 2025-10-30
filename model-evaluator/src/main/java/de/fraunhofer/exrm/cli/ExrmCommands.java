package de.fraunhofer.exrm.cli;

import de.fraunhofer.exrm.RunTool;
import picocli.CommandLine;

import java.util.concurrent.Callable;

@CommandLine.Command(name = "exrm", mixinStandardHelpOptions = true,
        version = "exrm-1.0", description = "")
public class ExrmCommands implements Callable<Integer> {

    @CommandLine.Option(names = {"-t", "--toolkit"}, description = {"ML toolkit: weka, meka, scikit"})
    private String mlToolkit = "meka";

    @CommandLine.Option(names = {"-c", "--config"}, description = {"Experiment configuration file"})
    private String configFile = "config.file";

    @Override
    public Integer call() throws Exception {
        return new RunTool().run(new ExrmConfigOptions(mlToolkit, configFile));
    }
}
