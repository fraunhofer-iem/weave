package de.fraunhofer.iem.weave.cli;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class PipelineOptions {

    /** Matches ${NAME} and ${NAME:-default} placeholders in property values. */
    private static final Pattern PLACEHOLDER =
            Pattern.compile("\\$\\{([^}:]+)(?::-([^}]*))?}");

    private String configFile;
    private String toolkit;
    private String dataset;
    private long timeout;
    private long nodeTimeout;
    private long candidateTimeout;
    private long seed;
    private int cpu;

    private String experimentConfigFile;
    private String dbConfigFile;
    private String  explainModel;
    private boolean explainerOnly;
    private String  testDatasetPath;

    private String pythonExplainerPath;
    private String outputPath;
    private String pythonPath;

    private int port;

    private int shapGlobalBgSamples;
    private int shapGlobalExpSamples;

    private int shapLocalBgSamples;
    private int shapLocalExpSamples;


    PipelineOptions() {
    }

    PipelineOptions(String toolkit, String configFile, String explainModel, boolean explainerOnly) {

        this.toolkit = toolkit;
        this.configFile = configFile;
        this.explainModel = explainModel;
        this.explainerOnly = explainerOnly;

        Properties properties = new Properties();

        try (InputStream is = Files.newInputStream(Path.of(configFile))) {
            properties.load(new InputStreamReader(is, StandardCharsets.UTF_8));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Paths may use ${VAR} placeholders and may be given relative to the config
        // file's own directory, so a config is not tied to one machine or one CWD.
        Path configDir = Path.of(configFile).toAbsolutePath().getParent();

        this.port = Integer.parseInt(properties.getProperty("port"));
        this.dataset = resolvePath(properties, "dataset.train", configDir);
        this.testDatasetPath = resolvePath(properties, "dataset.test", configDir);
        this.timeout = Long.parseLong(properties.getProperty("mlplan.timeout"));
        this.nodeTimeout = Long.parseLong(properties.getProperty("mlplan.timeout.node"));
        this.candidateTimeout = Long.parseLong(properties.getProperty("mlplan.timeout.candidate"));
        this.seed = Long.parseLong(properties.getProperty("mlplan.seed"));
        this.cpu = Integer.parseInt(properties.getProperty("mlplan.cpu"));

        this.dbConfigFile = resolvePath(properties, "jaicore.databaseConfig", configDir);
        this.experimentConfigFile = resolvePath(properties, "jaicore.experimentConfig", configDir);

        this.pythonExplainerPath = resolvePath(properties, "paths.explainer", configDir);
        this.outputPath = resolvePath(properties, "paths.output", configDir);
        // Not a path: this is an interpreter name ("python") as often as it is a path,
        // so it is expanded but never resolved against the config directory.
        this.pythonPath = expand(properties.getProperty("paths.python"), "paths.python");
        this.shapGlobalBgSamples = Integer.parseInt(properties.getProperty("shap.global.bg.samples"));
        this.shapGlobalExpSamples = Integer.parseInt(properties.getProperty("shap.global.exp.samples"));
        this.shapLocalBgSamples = Integer.parseInt(properties.getProperty("shap.local.bg.samples"));
        this.shapLocalExpSamples = Integer.parseInt(properties.getProperty("shap.local.exp.samples"));

    }

    /**
     * Substitutes every {@code ${NAME}} in the value with the environment variable
     * NAME, falling back to the system property of the same name. The shell-style
     * form {@code ${NAME:-default}} supplies a fallback when NAME is unset, which is
     * how per-run values such as MODEL_TAG stay optional.
     *
     * @throws IllegalStateException if a referenced variable is set nowhere and no
     *                               default is given, so a missing WEAVE_HOME fails
     *                               loudly at startup instead of producing a path
     *                               like "/evaluation/...".
     */
    private static String expand(String value, String key) {
        if (value == null) {
            return null;
        }
        Matcher matcher = PLACEHOLDER.matcher(value);
        StringBuilder result = new StringBuilder();
        while (matcher.find()) {
            String name = matcher.group(1);
            String fallback = matcher.group(2);
            String replacement = System.getenv(name);
            if (replacement == null) {
                replacement = System.getProperty(name);
            }
            if (replacement == null) {
                replacement = fallback;
            }
            if (replacement == null) {
                throw new IllegalStateException("Property '" + key + "' references ${" + name
                        + "}, but " + name + " is set neither in the environment nor as a"
                        + " system property (-D" + name + "=...).");
            }
            matcher.appendReplacement(result, Matcher.quoteReplacement(replacement));
        }
        matcher.appendTail(result);
        return result.toString();
    }

    /**
     * Reads a path-valued property: expands {@code ${NAME}} placeholders and, if the
     * result is relative, resolves it against the directory holding the config file
     * rather than the process working directory.
     */
    private static String resolvePath(Properties properties, String key, Path configDir) {
        String raw = properties.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return raw;
        }
        String expanded = expand(raw.trim(), key);
        Path path = Path.of(expanded);
        return path.isAbsolute() ? path.normalize().toString()
                : configDir.resolve(path).normalize().toString();
    }

    public String getConfigFile() {
        return configFile;
    }

    public void setConfigFile(String configFile) {
    }

    public String getToolkit() {
        return toolkit;
    }

    public void setToolkit(String toolkit) {
        this.toolkit = toolkit;
    }

    public String getDataset() {
        return dataset;
    }

    public void setDataset(String dataset) {
        this.dataset = dataset;
    }

    public long getTimeout() {
        return timeout;
    }

    public void setTimeout(long timeout) {
        this.timeout = timeout;
    }

    public long getSeed() {
        return seed;
    }

    public void setSeed(long seed) {
        this.seed = seed;
    }

    public int getCpu() {
        return cpu;
    }

    public void setCpu(int cpu) {
        this.cpu = cpu;
    }

    public String getExperimentConfigFile() {
        return experimentConfigFile;
    }

    public void setExperimentConfigFile(String experimentConfigFile) {
        this.experimentConfigFile = experimentConfigFile;
    }

    public String getDbConfigFile() {
        return dbConfigFile;
    }

    public void setDbConfigFile(String dbConfigFile) {
        this.dbConfigFile = dbConfigFile;
    }

    public long getNodeTimeout() {
        return nodeTimeout;
    }

    public void setNodeTimeout(long nodeTimeout) {
        this.nodeTimeout = nodeTimeout;
    }

    public long getCandidateTimeout() {
        return candidateTimeout;
    }

    public void setCandidateTimeout(long candidateTimeout) {
        this.candidateTimeout = candidateTimeout;
    }

    @Override
    public String toString() {
        return "PipelineOptions{" +
                "configFile='" + configFile + '\'' +
                ", toolkit='" + toolkit + '\'' +
                ", dataset='" + dataset + '\'' +
                ", timeout=" + timeout +
                ", nodeTimeout=" + nodeTimeout +
                ", candidateTimeout=" + candidateTimeout +
                ", seed=" + seed +
                ", cpu=" + cpu +
                ", testDatasetPath='" + testDatasetPath + '\'' +
                ", experimentConfigFile='" + experimentConfigFile + '\'' +
                ", dbConfigFile='" + dbConfigFile + '\'' +
                ", pythonExplainerPath='" + pythonExplainerPath + '\'' +
                ", outputPath='" + outputPath + '\'' +
                ", pythonPath='" + pythonPath + '\'' +
                '}';
    }

    public String getExplainModel() {
        return explainModel;
    }

    public void setExplainModel(String explainModel) {
        this.explainModel = explainModel;
    }

    public boolean isExplainerOnly() {
        return this.explainerOnly;
    }

    public String getTestDatasetPath() {
        return testDatasetPath;
    }

    public int getPort() {
        return port;
    }

    public String getPythonExplainerPath() {
        return pythonExplainerPath;
    }

    public String getOutputPath() {
        return outputPath;
    }

    public String getPythonPath() {
        return pythonPath;
    }

    public int getShapGlobalBgSamples() { return shapGlobalBgSamples; }
    public void setShapGlobalBgSamples(int shapGlobalBgSamples) { this.shapGlobalBgSamples = shapGlobalBgSamples; }
    public int getShapGlobalExpSamples() { return shapGlobalExpSamples; }
    public void setShapGlobalExpSamples(int shapGlobalExpSamples) { this.shapGlobalExpSamples = shapGlobalExpSamples; }

    public int getShapLocalBgSamples() { return shapLocalBgSamples; }
    public void setShapLocalBgSamples(int shapLocalBgSamples) { this.shapLocalBgSamples = shapLocalBgSamples; }
    public int getShapLocalExpSamples() { return shapLocalExpSamples; }
    public void setShapLocalExpSamples(int shapLocalExpSamples) { this.shapLocalExpSamples = shapLocalExpSamples; }
}
