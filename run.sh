#!/bin/bash
set -euo pipefail # Exit on error, undefined variable, or pipe failure

# --- Prerequisites ---
# Ensure you have the following installed before running:
# 1. bash
# 2. git
# 3. uv (https://github.com/astral-sh/uv)
# 4. A Python 3.9 interpreter accessible in your PATH or provide its full path below.
# 5. wget
# 6. tar
# 7. unzip
# 8. Required python scripts in 'sharegpt_translation/' directory: download_data.py, translate.py
# 9. Required 'inference/' directory sibling to 'sharegpt_translation/'

PYTHON_INTERPRETER="python3.9" # Or specify the full path, e.g., /usr/bin/python3.9

# --- Argument Handling ---
# Check if Hugging Face token was provided as argument
if [ -z "$1" ]; then
    echo "Usage: $0 <huggingface_token>"
    echo "Installs dependencies, downloads models, downloads data, and runs translation."
    echo "Get a token from https://huggingface.co/settings/tokens"
    echo "Example: $0 hf_YourTokenHere..."
    exit 1
fi
HF_TOKEN="$1"
echo "Using Hugging Face Token for downloads."


# --- Basic Setup ---
root_dir=$(pwd)
venv_dir=".venv" # Standard directory name for uv venvs

echo "Setting up the environment in $root_dir"
echo "Using Python interpreter: $PYTHON_INTERPRETER"

# --------------------------------------------------------------
#          Create the virtual environment with uv
# --------------------------------------------------------------
echo "Creating a virtual environment with $PYTHON_INTERPRETER"
# Check if the Python interpreter exists
if ! command -v $PYTHON_INTERPRETER &> /dev/null
then
    echo "Error: $PYTHON_INTERPRETER could not be found."
    echo "Please install Python 3.9 or specify the correct path in the script."
    exit 1
fi

# Check if venv directory already exists; if so, skip creation
if [ ! -d "$venv_dir" ]; then
    uv venv -p $PYTHON_INTERPRETER $venv_dir
    echo "Virtual environment created at $venv_dir"
else
    echo "Virtual environment directory '$venv_dir' already exists. Skipping creation."
    echo "If you need a fresh environment, please remove the '$venv_dir' directory first."
fi

echo "Installing all the dependencies using uv..."

# uv automatically uses the environment in the current dir (.venv)
# No need to upgrade pip separately, uv manages installations.

# --------------------------------------------------------------
#                   PyTorch Installation
# --------------------------------------------------------------
echo "Installing PyTorch..."
# Consider adding --no-cache-dir if experiencing caching issues
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 || { echo "PyTorch installation failed."; exit 1; }

# --------------------------------------------------------------
#       Install IndicNLP library and necessary resources
# --------------------------------------------------------------
echo "Cloning and installing IndicNLP library and resources..."
if [ ! -d "indic_nlp_resources" ]; then
    git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git || { echo "Failed to clone indic_nlp_resources."; exit 1; }
else
    echo "indic_nlp_resources already cloned."
fi
# Export for potential use within this script's context,
# but user needs to set it persistently if required outside the script.
export INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources
echo "Set INDIC_RESOURCES_PATH (for this script run): $INDIC_RESOURCES_PATH"
echo "Note: You might need to add 'export INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources' to your shell profile (~/.bashrc, ~/.zshrc) for persistent use."


# we use version 0.92 which is the latest in the github repo
if [ ! -d "indic_nlp_library" ]; then
    git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git || { echo "Failed to clone indic_nlp_library."; exit 1; }
else
    echo "indic_nlp_library already cloned."
fi

( # Run in subshell to isolate cd
  cd indic_nlp_library
  echo "Installing indic_nlp_library from local clone..."
  # Using --system allows uv pip install to work outside an activated venv context if needed, but usually not necessary if venv exists
  uv pip install --no-deps ./ || { echo "Failed to install indic_nlp_library."; exit 1; }
)
# cd $root_dir # No longer needed due to subshell

# --------------------------------------------------------------
#               Install additional utility packages
# --------------------------------------------------------------
echo "Installing utility packages..."

# --- Compatibility Fix for urduhack -> tf2crf -> tensorflow-addons ---
echo "Installing older TensorFlow version for compatibility with tensorflow-addons..."
# Ensure tensorflow-cpu is used if no GPU is intended/available, or adjust as needed
uv pip install "tensorflow-cpu<=2.10" || { echo "TensorFlow installation failed."; exit 1; } # Pin TF to a compatible version (e.g., 2.10.x)

# --- Now install the rest of the packages ---
echo "Installing nltk, urduhack, and other utilities..."
uv pip install \
    nltk \
    sacremoses \
    regex \
    pandas \
    mock \
    transformers==4.28.1 \
    sacrebleu==2.3.1 \
    urduhack[tf] \
    mosestokenizer \
    ctranslate2==3.9.0 \
    gradio || { echo "Utility package installation failed."; exit 1; }

echo "Downloading urduhack resources..."
# Use uv run to execute python within the managed environment
uv run python -c "import urduhack; urduhack.download()" || { echo "Urduhack download/import failed. Check TF compatibility or network connection."; exit 1; }


echo "Downloading nltk 'punkt' resource..."
uv run python -c "import nltk; nltk.download('punkt')" || { echo "NLTK punkt download failed."; exit 1; }

# --------------------------------------------------------------
#               Sentencepiece for tokenization
# --------------------------------------------------------------
echo "Installing Sentencepiece..."
uv pip install sentencepiece || { echo "Sentencepiece installation failed."; exit 1; }

# --------------------------------------------------------------
#               Fairseq Installation from Source
# --------------------------------------------------------------
echo "Cloning and installing Fairseq..."
if [ ! -d "fairseq" ]; then
    git clone https://github.com/pytorch/fairseq.git || { echo "Failed to clone fairseq."; exit 1; }
else
    echo "fairseq already cloned."
fi

( # Run in subshell
 cd fairseq
 echo "Installing fairseq from local clone..."
 uv pip install --no-deps ./ || { echo "Failed to install fairseq."; exit 1; }
)

# --------------------------------------------------------------
#       Download Model Files (Requires HF Token)
# --------------------------------------------------------------
echo "Downloading model distribution package from Hugging Face..."
wget --header="Authorization: Bearer $HF_TOKEN" -O en-indic-dist.tar.gz -c "https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-dist.tar.gz" || { echo "Failed to download en-indic-dist.tar.gz"; exit 1; }


echo "Extracting model distribution..."
tar -xvf en-indic-dist.tar.gz || { echo "Failed to extract en-indic-dist.tar.gz"; exit 1; }

echo "Cleaning up distribution tarball..."
rm -f en-indic-dist.tar.gz

echo "Creating checkpoints directory..."
mkdir -p checkpoints

echo "Moving extracted distribution files to checkpoints..."
# Ensure the source directory exists after extraction before moving
if [ -d "en-indic-dist" ]; then
    mv en-indic-dist/* checkpoints/ || { echo "Failed to move distribution files"; exit 1; }
else
    echo "Error: Expected directory 'en-indic-dist' not found after extraction."
    exit 1
fi

echo "Removing temporary distribution source directory..."
rm -rf en-indic-dist

# --- Download SPM ---
echo "Downloading SentencePiece model package from Hugging Face..."
wget --header="Authorization: Bearer $HF_TOKEN" -O en-indic-spm.zip -c "https://huggingface.co/datasets/ai4bharat/BPCC/resolve/main/additional/en-indic-spm.zip" || { echo "Failed to download en-indic-spm.zip"; exit 1; }

echo "Extracting SentencePiece models..."
# Use a temporary directory name to avoid potential clashes or confusion
temp_spm_extract_dir="en-indic-spm-extracted"
# Ensure the temp dir is clean before extraction
rm -rf "$temp_spm_extract_dir"
mkdir "$temp_spm_extract_dir"

unzip -o en-indic-spm.zip -d "$temp_spm_extract_dir" || { echo "Failed to unzip en-indic-spm.zip"; exit 1; }
rm -f en-indic-spm.zip

echo "Ensuring SentencePiece target directory exists..."
target_vocab_dir="./checkpoints/ct2_int8_model/vocab/"
mkdir -p "$target_vocab_dir"

# --- CORRECTED SECTION START ---
# Determine the actual source directory of the model files within the extracted archive.
# It might be directly in temp_spm_extract_dir or nested one level down (e.g., in temp_spm_extract_dir/en-indic-spm)
spm_source_dir="$temp_spm_extract_dir"
# Check if there's exactly one directory inside the extraction dir, which is a common pattern for archives containing a root folder.
potential_nested_dir=$(find "$temp_spm_extract_dir" -mindepth 1 -maxdepth 1 -type d)
# Use wc -l | xargs to get a clean count, works better if find output has leading/trailing spaces
num_nested_dirs=$(echo "$potential_nested_dir" | wc -l | xargs)

if [ "$num_nested_dirs" -eq 1 ] && [ -d "$potential_nested_dir" ]; then
    echo "Detected nested directory structure in SPM archive: $potential_nested_dir"
    # Check if model files exist within this nested directory
    if [ -f "$potential_nested_dir/model.SRC" ]; then
      spm_source_dir="$potential_nested_dir"
      echo "Using '$spm_source_dir' as the source for SPM files."
    else
       echo "Warning: Nested directory '$potential_nested_dir' found, but 'model.SRC' not directly inside it. Assuming files are in '$temp_spm_extract_dir'."
    fi
# Also handle the specific case observed if the generic check fails
elif [ -d "$temp_spm_extract_dir/en-indic-spm" ] && [ -f "$temp_spm_extract_dir/en-indic-spm/model.SRC" ]; then
    echo "Detected specific nested 'en-indic-spm' directory."
    spm_source_dir="$temp_spm_extract_dir/en-indic-spm"
    echo "Using '$spm_source_dir' as the source for SPM files."
elif [ ! -f "$temp_spm_extract_dir/model.SRC" ]; then
    # Check if the file exists anywhere one level deep
    found_model_src=$(find "$temp_spm_extract_dir" -mindepth 2 -maxdepth 2 -name 'model.SRC' | head -n 1)
    if [ -n "$found_model_src" ]; then
         echo "Warning: Found 'model.SRC' deeper than expected at '$found_model_src'. Adjusting move logic might be needed if this fails."
         echo "Attempting move assuming files are directly in '$temp_spm_extract_dir'."
         # Fallback to assuming files are directly in temp_spm_extract_dir; the move might fail later if this is wrong.
    else
        echo "Error: Cannot find 'model.SRC' directly in '$temp_spm_extract_dir' or in a single common nested directory pattern. Check the contents of the downloaded en-indic-spm.zip."
        exit 1
    fi
else
    echo "SPM files found directly in '$temp_spm_extract_dir'."
fi


echo "Moving SentencePiece model files from '$spm_source_dir' to '$target_vocab_dir'..."
# Move the *contents* of the determined source directory
# Use shopt -s dotglob to include hidden files if any, though unlikely needed for .SRC/.TGT
shopt -s dotglob
mv "$spm_source_dir"/* "$target_vocab_dir" || { echo "Failed to move SPM files from '$spm_source_dir' to '$target_vocab_dir'. Check permissions and paths."; exit 1; }
shopt -u dotglob # Turn off dotglob again
# --- CORRECTED SECTION END ---

echo "Removing temporary SentencePiece extraction directory..."
rm -rf "$temp_spm_extract_dir"


# --------------------------------------------------------------
#          Download Data and Run Translation Script
# --------------------------------------------------------------
sharegpt_dir="sharegpt_translation"
download_script="$sharegpt_dir/download_data.py"
translate_script="$sharegpt_dir/translate.py"
# Assuming download_data.py places this in the root_dir. Adjust if needed.
input_json_path="$root_dir/ShareGPT_V4.3_unfiltered_cleaned_split.json"
model_dir="$root_dir/checkpoints/ct2_int8_model"
inference_dir="$root_dir/inference" # Assuming the inference dir is also at root level

echo "Checking for required directories and scripts..."
if [ ! -d "$sharegpt_dir" ]; then
    echo "Error: Directory '$sharegpt_dir' not found in the current directory ($root_dir)."
    echo "Please ensure the '$sharegpt_dir' directory containing the python scripts exists."
    exit 1
fi
if [ ! -f "$download_script" ]; then
    echo "Error: Download script '$download_script' not found."
    exit 1
fi
if [ ! -f "$translate_script" ]; then
    echo "Error: Translation script '$translate_script' not found."
    exit 1
fi
if [ ! -d "$inference_dir" ]; then
    echo "Error: Directory '$inference_dir' not found in the current directory ($root_dir)."
    echo "This directory is required by '$translate_script' for imports."
    exit 1
fi

echo "Running data download script from $sharegpt_dir..."
# Execute the script using its path relative to the root_dir
uv run python "$download_script" || { echo "Failed to run $download_script"; exit 1; }

# Check for input file AFTER download script runs
if [ ! -f "$input_json_path" ]; then
    echo "Error: Input file '$input_json_path' not found in the root directory ($root_dir) after running download script."
    echo "Make sure '$download_script' creates it in the expected location."
    exit 1
fi

# Check for model directory just before translation
if [ ! -d "$model_dir" ]; then
    echo "Error: Model directory '$model_dir' not found. Check previous download and move steps."
    exit 1
fi
# Check specifically for the vocab directory within the model dir, as that's crucial
if [ ! -d "$model_dir/vocab" ]; then
    echo "Error: Vocab directory '$model_dir/vocab' not found. Check SPM download and move steps."
    exit 1
fi
# Check for the specific file that caused the original error
if [ ! -f "$model_dir/vocab/model.SRC" ]; then
    echo "Error: Expected file 'model.SRC' not found in '$model_dir/vocab/'. Check SPM download and move steps."
    ls -l "$model_dir/vocab/" # List contents for debugging
    exit 1
fi


echo "Running translation script from $sharegpt_dir..."
# --- Execute translate.py with adjusted PYTHONPATH ---
# Add the root directory to PYTHONPATH so translate.py can find the sibling 'inference' directory
echo "Executing $translate_script with PYTHONPATH set to $root_dir"
PYTHONPATH="$root_dir" uv run python "$translate_script" -i "$input_json_path" -m "$model_dir" || {
    echo "-----------------------------------------------------"
    echo "ERROR: Failed to run $translate_script"
    echo "This might be due to Python import errors or issues within the script itself."
    echo "Check that the 'inference' directory exists at $root_dir"
    echo "and that $translate_script can import modules from it."
    echo "Also check the script's internal logic and previous logs for errors."
    echo "-----------------------------------------------------"
    exit 1
}

echo "Translation script finished successfully."

# --------------------------------------------------------------
#                       Final Instructions
# --------------------------------------------------------------
echo ""
echo "--------------------------------------------------------------"
echo "                 Setup and Execution Complete                 "
echo "--------------------------------------------------------------"
echo ""
echo "Model files are in: $root_dir/checkpoints"
echo "SPM vocabulary files are in: $model_dir/vocab"
echo "Input data file used: $input_json_path"
echo "Translation process has been run."
echo ""
echo "To work interactively within the environment, activate it using:"
echo "source $venv_dir/bin/activate"
echo ""
echo "Remember to set the INDIC_RESOURCES_PATH environment variable if needed for your workflow:"
echo "export INDIC_RESOURCES_PATH=$root_dir/indic_nlp_resources"
echo "(You might want to add this export line to your ~/.bashrc or ~/.zshrc)"

exit 0