#!/bin/bash
# Creates the environment to run eXRM locally

set -e

# Set the virtual environment directory
VENV_DIR="./venv"

# Create the virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"


# Upgrade pip and install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r shap/weka/requirements.txt
pip install -r shap/meka/requirements.txt

echo "Venv created successfully..."