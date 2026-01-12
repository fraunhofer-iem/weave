package de.fraunhofer.iem.weave.module.explainer.server;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import de.fraunhofer.iem.weave.module.explainer.services.PredictionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;

/**
 * A lightweight HTTP server that exposes the /predict endpoint and invokes a PredictionService
 * instance.
 */
public class PredictionHttpServer {

    private static final Logger logger = LoggerFactory.getLogger(PredictionHttpServer.class);

    private final PredictionService predictor;
    private final int port;
    private HttpServer server;
    private final ObjectMapper mapper = new ObjectMapper();

    /**
     * Creates a new HTTP prediction server instance.
     *
     * @param predictor the PredictionService used
     * @param port      the port on which the HTTP server should listen.
     */
    public PredictionHttpServer(PredictionService predictor, int port) {
        this.predictor = predictor;
        this.port = port;
    }

    /**
     * Starts the underlying HttpServer.

     */
    public void start() throws Exception {
        server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/predict", new PredictHandler());
        server.setExecutor(null);
        server.start();
        logger.info("Prediction HTTP server started on port {}", port);
    }

    /**
     * Stops the HTTP server.
     *
     * @param delaySeconds the maximum time in seconds to wait.
     */
    public void stop(int delaySeconds) {
        if (server != null) {
            server.stop(delaySeconds);
            logger.info("Prediction HTTP server stopped.");
        }
    }

    /**
     * HttpHandler that processes POST requests to the /predict endpoint.
     */
    private class PredictHandler implements HttpHandler {

        /**
         * Handles incoming HTTP requests to the /predict endpoint.
         *
         * @param exchange the HttpExchange of the HTTP request
         *                 and response.
         */
        @Override
        @SuppressWarnings("unchecked")
        public void handle(HttpExchange exchange) {
            try {
                if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1); // Method Not Allowed
                    return;
                }

                // Parse request body
                try (InputStream is = exchange.getRequestBody()) {
                    Map<String, Object> body = mapper.readValue(is, Map.class);
                    Object instancesObj = body.get("instances");
                    if (!(instancesObj instanceof List)) {
                        sendBadRequest(exchange, "\"instances\" field must be a list");
                        return;
                    }

                    List<List<Object>> instances = (List<List<Object>>) instancesObj;

                    // Convert to double[][]
                    double[][] features = new double[instances.size()][];
                    for (int i = 0; i < instances.size(); i++) {
                        List<Object> row = instances.get(i);
                        features[i] = new double[row.size()];
                        for (int j = 0; j < row.size(); j++) {
                            Object value = row.get(j);
                            if (value instanceof Number) {
                                features[i][j] = ((Number) value).doubleValue();
                            } else {
                                features[i][j] = Double.parseDouble(value.toString());
                            }
                        }
                    }

                    // Call prediction service
                    double[][] probs = predictor.predictProba(features);

                    // Build response
                    String responseJson = mapper.writeValueAsString(Map.of("probs", probs));
                    byte[] respBytes = responseJson.getBytes(StandardCharsets.UTF_8);

                    exchange.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
                    exchange.sendResponseHeaders(200, respBytes.length);
                    try (OutputStream os = exchange.getResponseBody()) {
                        os.write(respBytes);
                    }
                }

            } catch (Exception e) {
                logger.error("Error handling /predict request", e);
                try {
                    exchange.sendResponseHeaders(500, -1);
                } catch (Exception ignored) {
                    logger.error("Error sending error code", e);
                }
            }
        }

        private void sendBadRequest(HttpExchange exchange, String message) {
            try {
                byte[] respBytes = message.getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().add("Content-Type", "text/plain; charset=utf-8");
                exchange.sendResponseHeaders(400, respBytes.length);
                try (OutputStream os = exchange.getResponseBody()) {
                    os.write(respBytes);
                }
            } catch (Exception e) {
                logger.error("Error sending 400 Bad Request", e);
            }
        }
    }
}

