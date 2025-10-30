package de.fraunhofer.exrm;

import de.fraunhofer.exrm.cli.ExrmCommands;
import picocli.CommandLine;

public class Main {

    public static void main(String[] args) {
        new CommandLine(new ExrmCommands()).execute(args);
    }
}
