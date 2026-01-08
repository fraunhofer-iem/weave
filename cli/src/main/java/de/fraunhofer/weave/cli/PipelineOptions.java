package de.fraunhofer.weave.cli;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

public class PipelineOptions {

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

    private int port;

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

        this.port = Integer.parseInt(properties.getProperty("port"));
        this.dataset = properties.getProperty("dataset.train");
        this.testDatasetPath = properties.getProperty("dataset.test");
        this.timeout = Long.parseLong(properties.getProperty("mlplan.timeout"));
        this.nodeTimeout = Long.parseLong(properties.getProperty("mlplan.timeout.node"));
        this.candidateTimeout = Long.parseLong(properties.getProperty("mlplan.timeout.candidate"));
        this.seed = Long.parseLong(properties.getProperty("mlplan.seed"));
        this.cpu = Integer.parseInt(properties.getProperty("mlplan.cpu"));

        this.dbConfigFile = properties.getProperty("jaicore.databaseConfig");
        this.experimentConfigFile = properties.getProperty("jaicore.experimentConfig");

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
                ", experimentConfigFile='" + experimentConfigFile + '\'' +
                ", dbConfigFile='" + dbConfigFile + '\'' +
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
}
