package de.fraunhofer.iem.weave;

import de.fraunhofer.iem.weave.cli.CliOptions;
import picocli.CommandLine;

public class Main {

    public static void main(String[] args) {
        new CommandLine(new CliOptions()).execute(args);
    }
}
