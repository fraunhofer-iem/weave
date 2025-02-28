import iem.fraunhofer.xai_srm.dataset.json.Dataset;
import iem.fraunhofer.xai_srm.dataset.json.DatasetLoader;
import org.junit.jupiter.api.Test;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

class LoadJsonDatasetTest {

    @Test
    void loadDatasetFile() {

        Dataset dataset;

        try {
            DatasetLoader datasetLoader = new DatasetLoader();
            dataset = datasetLoader.importFile("/home/oshando/IdeaProjects/xai-srm/srm-dataset/src/main/resources/srm-dataset.json");

        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        assertEquals(1885, dataset.getMethods().size());
    }
}