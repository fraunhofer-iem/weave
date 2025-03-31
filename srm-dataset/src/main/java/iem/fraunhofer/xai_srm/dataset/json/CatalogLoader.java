package iem.fraunhofer.xai_srm.dataset.json;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

/**
 * Helpers to import and export JSON dataset files.
 */
public class CatalogLoader {

    /**
     * Imports SRMs from JSON file.
     *
     * @param file JSON File that stores security-relevant methods
     * @return object containing all security-relevant methods
     */
    public Catalog importFile(String file) throws IOException {

        ObjectMapper objectMapper = new ObjectMapper();

        Catalog dataset = objectMapper.readValue(new File(file), Catalog.class);

        return dataset;
    }

    /**
     * Exports SRM list to JSON file.
     *
     * @param srmList list of SRMa
     * @param file    path of JSON file
     */
    public void exportFile(Catalog srmList, String file) throws IOException {

        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(new File(file), srmList);
    }
}
