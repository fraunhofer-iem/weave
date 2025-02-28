package iem.fraunhofer.xai_srm.dataset.json;

/**
 * Returns an artifact object containing the name of the compiled and source JARs/file
 * and the Maven identifier for the project.
 */
public class Artifacts {

    private String compiled;
    private String sources;
    private String identifier;

    public String getCompiled() {
        return compiled;
    }

    public void setCompiled(String compiled) {
        this.compiled = compiled;
    }

    public String getSources() {
        return sources;
    }

    public void setSources(String sources) {
        this.sources = sources;
    }

    public String getIdentifier() {
        return identifier;
    }

    public void setIdentifier(String identifier) {
        this.identifier = identifier;
    }
}
