package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;
import java.util.Set;

/**
 * Returns method object containing the JSON properties in the dataset file.
 */
public class Method {

    private String name;
    private List<String> parameters;
    private String signature;
    private String framework;
    private String link;
    private String comment;
    private String discovery;
    private DataFlow dataIn;
    private DataFlow dataOut;
    private Set<Category> srm;
    private Set<Category> cwe;
    private boolean known;
    private String body;
    private Javadoc javadoc;
    @JsonProperty("return")
    private String returnType;
    @JsonProperty("interface")
    private boolean isInterface;
    private Artifacts artifacts;
    private List<Usage> usages;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public List<String> getParameters() {
        return parameters;
    }

    public void setParameters(List<String> parameters) {
        this.parameters = parameters;
    }

    public String getSignature() {
        return signature;
    }

    public void setSignature(String signature) {
        this.signature = signature;
    }

    public String getFramework() {
        return framework;
    }

    public void setFramework(String framework) {
        this.framework = framework;
    }

    public String getLink() {
        return link;
    }

    public void setLink(String link) {
        this.link = link;
    }

    public String getComment() {
        return comment;
    }

    public void setComment(String comment) {
        this.comment = comment;
    }

    public String getDiscovery() {
        return discovery;
    }

    public void setDiscovery(String discovery) {
        this.discovery = discovery;
    }

    public DataFlow getDataIn() {
        return dataIn;
    }

    public void setDataIn(DataFlow dataIn) {
        this.dataIn = dataIn;
    }

    public DataFlow getDataOut() {
        return dataOut;
    }

    public void setDataOut(DataFlow dataOut) {
        this.dataOut = dataOut;
    }

    public Set<Category> getSrm() {
        return srm;
    }

    public void setSrm(Set<Category> srm) {
        this.srm = srm;
    }

    public Set<Category> getCwe() {
        return cwe;
    }

    public void setCwe(Set<Category> cwe) {
        this.cwe = cwe;
    }

    public boolean isKnown() {
        return known;
    }

    public void setKnown(boolean known) {
        this.known = known;
    }

    public String getBody() {
        return body;
    }

    public void setBody(String body) {
        this.body = body;
    }

    public Javadoc getJavadoc() {
        return javadoc;
    }

    public void setJavadoc(Javadoc javadoc) {
        this.javadoc = javadoc;
    }

    public String getReturnType() {
        return returnType;
    }

    public void setReturnType(String returnType) {
        this.returnType = returnType;
    }

    public boolean isInterface() {
        return isInterface;
    }

    public void setInterface(boolean anInterface) {
        isInterface = anInterface;
    }

    public Artifacts getArtifacts() {
        return artifacts;
    }

    public void setArtifacts(Artifacts artifacts) {
        this.artifacts = artifacts;
    }

    public List<Usage> getUsages() {
        return usages;
    }

    public void setUsages(List<Usage> usages) {
        this.usages = usages;
    }
}
