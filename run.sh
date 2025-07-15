#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Run the Python script with all passed arguments
python openimagegen/main.py "$@"
