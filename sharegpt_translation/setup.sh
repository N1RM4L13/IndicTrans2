#!/bin/bash

# Check if token was provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <huggingface_token>"
    echo "Example: $0 hf_IpMglmOu..."
    exit 1
fi

# Set token from command line argument
HF_TOKEN="$1"

# Download the distribution package
wget --header="Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-dist.tar.gz"

# Extract the tar file
tar -xvf en-indic-dist.tar.gz

# Remove the tar file
rm -rf en-indic-dist.tar.gz

# Create checkpoints directory
mkdir -p checkpoints

# Move extracted files to checkpoints
mv en-indic-dist/* checkpoints/

# Remove the source directory
rm -rf en-indic-dist

# Download the SPM package
wget --header="Authorization: Bearer $HF_TOKEN" "https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-spm.zip"

# Extract the zip file
unzip en-indic-spm.zip

# Remove the zip file
rm -rf en-indic-spm.zip

# Ensure the target directory exists
mkdir -p ./checkpoints/ct2_int8_model/vocab/

# Move SPM files to the appropriate directory
mv en-indic-spm/* ./checkpoints/ct2_int8_model/vocab/

# Remove the source directory
rm -rf en-indic-spm