FROM python:3.10-slim

# install venv and java
RUN apt-get update && apt-get install -y \
    python3-venv \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/* \

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy datasets and scripts
COPY evaluation /app/evaluation
COPY shap /app/shap
COPY RQ2_and_RQ3.sh .

# Install dependencies
RUN pip install --upgrade pip && pip install -r shap/weka/requirements.txt
RUN pip install -r shap/meka/requirements.txt

# Provide file permissions
RUN chmod +x RQ2_and_RQ3.sh


CMD ["./RQ2_and_RQ3.sh"]



