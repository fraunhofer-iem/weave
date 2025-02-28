package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

/**
 * Helpers to import and export JSON dataset files.
 */
public class DatasetLoader {

    /**
     * Imports SRMs from JSON file.
     *
     * @param file JSON File that stores security-relevant methods
     * @return object containing all security-relevant methods
     */
    public Dataset importFile(String file) throws IOException {

        ObjectMapper objectMapper = new ObjectMapper();

        Dataset dataset = objectMapper.readValue(new File(file), Dataset.class);

        return dataset;
    }

    /**
     * Exports SRM list to JSON file.
     *
     * @param srmList list of SRMa
     * @param file    path of JSON file
     */
    public void exportFile(Dataset srmList, String file) throws IOException {

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(file), srmList);
    }
}
