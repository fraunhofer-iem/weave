package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.annotation.JsonValue;

/**
 * Eum for the SRM and CWE categories.
 */
public enum Category {

    SOURCE("source"),
    SINK("sink"),
    SANITIZER("sanitizer"),
    AUTHENTICATION_TO_HIGH("auth-safe-state"),
    AUTHENTICATION_TO_LOW("auth-unsafe-state"),
    AUTHENTICATION_NEUTRAL("auth-no-change"),
    PROPAGATOR("propagator"),
    CWE22("CWE22"),
    CWE35("CWE35"),
    CWE77("CWE77"),
    CWE78("CWE78"),
    CWE79("CWE79"),
    CWE89("CWE89"),
    CWE90("CWE90"),
    CWE91("CWE91"),
    CWE117("CWE117"),
    CWE233("CWE233"),
    CWE306("CWE306"),
    CWE327("CWE327"),
    CWE328("CWE328"),
    CWE443("CWE443"),
    CWE501("CWE501"),
    CWE601("CWE601"),
    CWE643("CWE643"),
    CWE862("CWE862"),
    CWE863("CWE863"),
    CWE917("CWE917"),
    CWE918("CWE918");

    private final String id;

    Category(String id) {
        this.id = id;
    }

    @JsonValue
    public String getId() {
        return id;
    }


}
