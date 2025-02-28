package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.ArrayList;
import java.util.List;

/**
 * Returns data-flow object that can represent data-in and data-out data flow operations.
 */
public class DataFlow {

    @JsonProperty("return")
    private boolean returnValue = false;
    private List<Integer> parameters = new ArrayList<Integer>();

    public DataFlow(boolean rT, List<Integer> parInd) {
        setReturnValue(rT);
        setParameters(parInd);
    }

    public DataFlow() {

    }

    public List<Integer> getParameters() {
        return parameters;
    }

    public void setParameters(List<Integer> parameters) {
        this.parameters = parameters;
    }

    public boolean getReturnValue() {
        return returnValue;
    }

    public void setReturnValue(boolean returnValue) {
        this.returnValue = returnValue;
    }

}
