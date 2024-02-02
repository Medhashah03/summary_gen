#!/bin/bash

# Ensure the script has executable permissions
chmod +x ./scripts/build.sh

# Install Python dependencies
pip install -r requirements.txt

# Execute the build script
./scripts/build.sh
