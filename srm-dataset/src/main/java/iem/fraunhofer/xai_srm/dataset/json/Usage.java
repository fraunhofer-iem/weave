package iem.fraunhofer.xai_srm.dataset.json;

/**
 * Returns object containing usages of a method.
 */
public class Usage {

    String path;
    int lineNumber;

    public String getPath() {
        return path;
    }

    public void setPath(String path) {
        this.path = path;
    }

    public int getLineNumber() {
        return lineNumber;
    }

    public void setLineNumber(int lineNumber) {
        this.lineNumber = lineNumber;
    }

}
