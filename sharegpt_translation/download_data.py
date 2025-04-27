
import requests
import os

# File URL and output path
file_url = "https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V4.3_unfiltered_cleaned_split.json"
output_file = "ShareGPT_V4.3_unfiltered_cleaned_split.json"

# Add your Hugging Face token if the repository is private
headers = {
    # "Authorization": "Bearer YOUR_HF_TOKEN"  
}

print(f"Downloading {output_file}...")
response = requests.get(file_url, headers=headers)

if response.status_code == 200:
    with open(output_file, "wb") as f:
        f.write(response.content)
    print(f"Successfully saved to {os.path.abspath(output_file)}")
else:
    print(f"Failed to download: {response.status_code} {response.reason}")