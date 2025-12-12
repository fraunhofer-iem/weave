package de.fraunhofer.weave.cli.module;

import de.fraunhofer.weave.cli.PipelineOptions;

public class ModelExplainer {

    private PipelineOptions options;
    private String classifier;

    public ModelExplainer(PipelineOptions options, String classifier) {
        this.options = options;
        this.classifier = classifier;
    }

    public void explain() {


    }

}
