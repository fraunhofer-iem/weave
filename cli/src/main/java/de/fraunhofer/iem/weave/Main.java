package de.fraunhofer.iem.weave;

import de.fraunhofer.iem.weave.cli.CliOptions;
import picocli.CommandLine;

public class Main {

    public static void main(String[] args) {
        // Propagate the pipeline's exit code: it was previously discarded, so a failed
        // run still looked successful to the batch scheduler. Exiting explicitly also
        // guarantees termination if any non-daemon thread is still lingering.
        System.exit(new CommandLine(new CliOptions()).execute(args));
    }
}
