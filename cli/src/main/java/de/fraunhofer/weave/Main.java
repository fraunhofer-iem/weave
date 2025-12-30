package de.fraunhofer.weave;

import de.fraunhofer.weave.cli.CliOptions;
import picocli.CommandLine;

public class Main {

    public static void main(String[] args) {
        new CommandLine(new CliOptions()).execute(args);
    }
}
