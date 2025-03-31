package iem.fraunhofer.xai_srm.dataset.json;

import java.util.Set;

/**
 * Returns dataset object containing a set of the methods and dataset metadata.
 */
public class Catalog {

    private Set<Method> methods;
    private String version;

    public Set<Method> getMethods() {
        return methods;
    }

    public void setMethods(Set<Method> methods) {
        this.methods = methods;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }
}
