FROM eclipse-temurin:21-jdk

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python || true

WORKDIR /app

COPY target/cli-1.0-jar-with-dependencies.jar /app/weave.jar
COPY scripts/shap_explainer.py /app/scripts/shap_explainer.py
COPY data /app/data

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install \
        matplotlib==3.10.0 \
        numpy==2.1.3 \
        pandas==2.2.3 \
        scikit-learn==1.6.1 \
        scipy==1.15.1 \
        shap==0.46.0 \
        requests

RUN mkdir -p /app/out

ENV PYTHON_EXECUTABLE=python
ENV SHAP_SCRIPT=/app/scripts/shap_explainer.py
ENV SHAP_OUTPUT_DIR=/app/out

EXPOSE 8090

ENTRYPOINT ["/bin/bash"]