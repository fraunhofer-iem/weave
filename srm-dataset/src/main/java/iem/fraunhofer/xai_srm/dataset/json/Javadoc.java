package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Returns doc comments object for class/method.
 */
public class Javadoc {

    @JsonProperty("method")
    private String methodComment;
    @JsonProperty("class")
    private String classComment;

    public Javadoc() {

        methodComment = "";
        classComment = "";
    }

    public Javadoc(String methodComment, String classComment) {
        this.methodComment = methodComment;
        this.classComment = classComment;
    }

    public String getMethodComment() {
        return methodComment;
    }

    public void setMethodComment(String methodComment) {
        this.methodComment = methodComment;
    }

    public String getClassComment() {
        return classComment;
    }

    public void setClassComment(String classComment) {
        this.classComment = classComment;
    }

    @JsonIgnore
    public String getMergedComments() {
        return methodComment + " " + classComment;
    }

    @Override
    public String toString() {
        return "Javadoc{" +
                "methodComment='" + methodComment + '\'' +
                ", classComment='" + classComment + '\'' +
                '}';
    }
}
