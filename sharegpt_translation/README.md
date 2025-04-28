## Installation

```bash
# Clone the repository
git clone https://github.com/N1RM4L13/IndicTrans2.git

# Enter the repository directory
cd IndicTrans2

# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

## Setting up translation

```bash
# Navigate to the translation directory
cd sharegpt_translation/

# Make the setup script executable
chmod +x setup.sh

# Run the setup script with your Hugging Face token
./setup.sh <huggingface_token>
```

## Data preparation and translation

```bash
# Download the necessary data
python3 download_data.py

# Run the translation
python3 translate.py -i ShareGPT_V4.3_unfiltered_cleaned_split.json -m checkpoints/ct2_int8_model
```
