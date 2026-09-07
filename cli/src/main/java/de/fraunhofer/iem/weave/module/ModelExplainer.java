package de.fraunhofer.iem.weave.module;


import de.fraunhofer.iem.weave.cli.PipelineOptions;
import de.fraunhofer.iem.weave.module.explainer.server.PredictionHttpServer;
import de.fraunhofer.iem.weave.module.explainer.services.ExplainerExporter;
import de.fraunhofer.iem.weave.module.explainer.services.PredictionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates the Java-side of the explainability pipeline using the HTTP endpoint.
 */
public class ModelExplainer {

    private static final Logger logger = LoggerFactory.getLogger(ModelExplainer.class);

    private final PredictionService predictor;
    private final PipelineOptions options;
    private PredictionHttpServer server;

    /**
     * Creates a new Explainer using the given prediction service and configuration.
     *
     */
    public ModelExplainer(PredictionService predictor,
                     PipelineOptions options) {
        this.predictor = predictor;
        this.options = options;
    }

    /**
     * Runs the HTTP-based explainability setup:
     */
    public void runHttpExplainer() throws Exception {
        // 1. Start the HTTP prediction server.
        this.server = new PredictionHttpServer(predictor, options.getPort());
        server.start();

        // 2. Export global and local feature CSVs.
        ExplainerExporter exporter = new ExplainerExporter(options);
        Path globalPath = exporter.exportGlobalFeatures();
        Path localPath = exporter.exportLocalFeatures();

        logger.info("HTTP explainer setup complete.");
        logger.info("Prediction server listening on http://localhost:{}/predict", server.getPort());
        logger.info("Global features CSV: {}", globalPath.toAbsolutePath());
        logger.info("Local features CSV : {}", localPath.toAbsolutePath());
        logger.info("You can now run the Python SHAP script pointing to these files and the /predict endpoint.");
    }

    /**
     * Launches the external Python SHAP script as a separate process.
     * @param pythonExecutable path or name of the Python interpreter to use (e.g., "python" or "C:\\Users\\..\\...\\python.exe").
     * @param shapScriptPath   path to the shap_http_explainer.py script.
     * @param outputDir        directory where SHAP plots should be written.
     */
    public void runShapExplainer(String pythonExecutable, String shapScriptPath, String outputDir, int shapGlobalBgSamples,
                                 int shapGlobalExpSamples, int shapLocalBgSamples, int shapLocalExpSamples) throws Exception {

        String toolkit = options.getToolkit();
        // Read the bound port from the server, not the config: with port 0 the OS picks it.
        String serverUrl = "http://localhost:" + server.getPort();

        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(shapScriptPath);
        command.add("--toolkit");
        command.add(toolkit);
        command.add("--server_url");
        command.add(serverUrl);
        command.add("--global_csv");
        command.add(Paths.get(options.getOutputPath(),"global_features.csv").toString());
        command.add("--local_csv");
        command.add(Paths.get(options.getOutputPath(),"local_features.csv").toString());
        command.add("--output_dir");
        command.add(outputDir);
        command.add("--shap_global_bg_samples");
        command.add(String.valueOf(shapGlobalBgSamples));
        command.add("--shap_global_exp_samples");
        command.add(String.valueOf(shapGlobalExpSamples));
        command.add("--shap_local_bg_samples");
        command.add(String.valueOf(shapLocalBgSamples));
        command.add("--shap_local_exp_samples");
        command.add(String.valueOf(shapLocalExpSamples));

        // Optional: lets the explainer name its per-class output. Skipped when a name
        // contains a comma, which would be indistinguishable from the separator - the
        // explainer then falls back to class indices.
        List<String> outputNames = predictor.getOutputNames();
        if (outputNames.stream().anyMatch(name -> name.contains(","))) {
            logger.warn("Not passing class names to the explainer: {} contains a comma.",
                    outputNames);
        } else if (!outputNames.isEmpty()) {
            command.add("--class_names");
            command.add(String.join(",", outputNames));
        }

        logger.info("Starting Python SHAP script: {}", String.join(" ", command));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true); // merge stdout and stderr

        Process process = pb.start();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                logger.info("[shap] {}", line);
            }
        }

        int exitCode = process.waitFor();
        if (exitCode != 0) {
            throw new IllegalStateException(
                    "Python SHAP script exited with non-zero status: " + exitCode);
        }

        logger.info("Python SHAP script completed successfully, plots should be in: {}", outputDir);
    }

    /**
     * Stops the prediction HTTP server if it is running.
     */
    public void stopServer(int delaySeconds) {
        if (server != null) {
            server.stop(delaySeconds);
        }
    }
}
