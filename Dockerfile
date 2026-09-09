# Container image for WEAVE and the ML4SRM replication package.
#
# The image copies the assembled jar, it does not build it. From the repository
# root, which is also the build context:
#
#   mvn -q clean package
#   docker build -t weave .
#   docker run --rm -it weave
#
# Inside the container the invocation is identical to the one on the host
# (see evaluation/ml4srm/README.md):
#
#   MODEL_TAG=new java -jar /app/weave.jar -t weka \
#     -c /app/evaluation/ml4srm/sscm/sscm.properties \
#     -X -e "<model descriptor from evaluation/ml4srm/sscm.sh>"
#
# Results land in /app/evaluation/ml4srm/<case>/explanations/<MODEL_TAG>, so
# mount a volume there to keep them.

FROM eclipse-temurin:21-jdk

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    bash \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The image mirrors the repository layout under /app so that WEAVE_HOME=/app
# makes every ${WEAVE_HOME} placeholder in the committed .properties files
# resolve without editing them.
ENV WEAVE_HOME=/app

# Dependencies first, so this layer is only rebuilt when requirements.txt
# changes. Installed from the same pinned file the cluster jobs use, so the
# container and the evaluation/ml4srm runs produce the same explanations.
# Keep it as one pip invocation - see the notes in requirements.txt for why
# splitting it apart breaks the numba/NumPy combination.
COPY evaluation/ml4srm/requirements.txt /app/evaluation/ml4srm/requirements.txt
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r /app/evaluation/ml4srm/requirements.txt \
 && python -c "import shap, numpy, pandas, matplotlib, requests"

COPY cli/target/cli-1.0-jar-with-dependencies.jar \
     /app/cli/target/cli-1.0-jar-with-dependencies.jar

# paths.explainer in every .properties file points at the script's location in
# the source tree, so it has to keep that path inside the image too.
COPY cli/src/main/java/de/fraunhofer/iem/weave/module/explainer/shap/shap_explainer.py \
     /app/cli/src/main/java/de/fraunhofer/iem/weave/module/explainer/shap/shap_explainer.py

# Datasets, .properties, experiments.cnf and db.properties for the seven case
# studies. The committed SHAP results and job logs are excluded by
# .dockerignore - the container regenerates them.
COPY evaluation/ml4srm /app/evaluation/ml4srm

RUN ln -s /app/cli/target/cli-1.0-jar-with-dependencies.jar /app/weave.jar

ENTRYPOINT ["/bin/bash"]
