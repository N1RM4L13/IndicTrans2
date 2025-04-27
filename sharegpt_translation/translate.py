import json
import time
import sys
import os
from tqdm import tqdm

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

try:
    from inference.engine import Model
except ImportError:
    print("Failed to import Model from inference.engine. Check your path configuration.")
    sys.exit(1)

def load_json_data(file_path):
    """Load JSON data from the given file path."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        sys.exit(1)

def initialize_model(model_path, device="cpu"):
    """Initialize the translation model."""
    try:
        start_time = time.time()
        model = Model(model_path, model_type="ctranslate2", device=device)
        end_time = time.time()
        print(f"Model loaded in {end_time - start_time:.2f} seconds")
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

def translate_conversation(model, conversation, src_lang="eng_Latn", tgt_lang="hin_Deva", 
                           progress_bar=None):
    """
    Translate all messages in a conversation.
    Returns a new conversation with translations added.
    
    Args:
        model: The translation model
        conversation: List of messages to translate
        src_lang: Source language code
        tgt_lang: Target language code
        progress_bar: tqdm progress bar to update (instead of creating a new one)
    """
    translated_conversation = []
    
    for msg in conversation:
        translated_msg = msg.copy()  # Create a copy to avoid modifying the original
        
        if msg.get('value'):
            # Split the message content into sentences for batch translation
            sentences = [line.strip() for line in msg['value'].split('\n') if line.strip()]
            
            if len(sentences) > 0:
                # Process in batches for efficiency
                translations = model.batch_translate(sentences, src_lang, tgt_lang)
                
                # Combine translated sentences back into a paragraph/message
                translated_msg['translated_value'] = '\n'.join(translations)
            else:
                translated_msg['translated_value'] = ""
        
        translated_conversation.append(translated_msg)
        
        # Update the progress bar if provided
        if progress_bar is not None:
            progress_bar.update(1)
    
    return translated_conversation

def process_file(json_file_path, model_path, output_file_path=None, 
                 src_lang="eng_Latn", tgt_lang="hin_Deva", device="cpu"):
    """
    Process a single JSON file containing conversations.
    """
    # Load the data
    data = load_json_data(json_file_path)
    
    # Initialize the translation model
    model = initialize_model(model_path, device)
    
    # Determine if it's a list of conversations or a single conversation
    if isinstance(data, list):
        # Multiple conversations
        print(f"Processing {len(data)} conversations")
        all_translated = []
        
        # Calculate total number of messages across all conversations for progress tracking
        total_messages = sum(len(item.get('conversations', [])) for item in data)
        
        # Create a single progress bar for all messages
        with tqdm(total=total_messages, desc="Translating all messages", unit="msg") as pbar:
            for conv_item in data:
                if 'conversations' in conv_item:
                    # Translate the conversation, passing the progress bar
                    translated_conv = translate_conversation(
                        model, conv_item['conversations'], src_lang, tgt_lang, progress_bar=pbar)
                    
                    # Create a new item with the translated conversation
                    new_item = conv_item.copy()
                    new_item['conversations'] = translated_conv
                    all_translated.append(new_item)
                else:
                    print(f"Warning: Item without 'conversations' key found: {conv_item.get('id', 'unknown')}")
                    all_translated.append(conv_item)
        
        # Save the translated data
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_translated, f, ensure_ascii=False, indent=2)
            print(f"Saved translated data to {output_file_path}")
        
        return all_translated
    
    elif isinstance(data, dict) and 'conversations' in data:
        # Single conversation
        print(f"Processing single conversation with ID: {data.get('id', 'unknown')}")
        
        # Create a single progress bar for all messages in this conversation
        total_messages = len(data['conversations'])
        with tqdm(total=total_messages, desc="Translating messages", unit="msg") as pbar:
            # Translate the conversation, passing the progress bar
            translated_conv = translate_conversation(
                model, data['conversations'], src_lang, tgt_lang, progress_bar=pbar)
        
        # Create a new item with the translated conversation
        new_data = data.copy()
        new_data['conversations'] = translated_conv
        
        # Save the translated data
        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            print(f"Saved translated data to {output_file_path}")
        
        return new_data
    
    else:
        print("Error: Unsupported data format")
        return None

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Translate ShareGPT conversations')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--model', '-m', required=True, help='Path to the translation model')
    parser.add_argument('--src-lang', default='eng_Latn', help='Source language code')
    parser.add_argument('--tgt-lang', default='hin_Deva', help='Target language code')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device to run the model on')
    
    args = parser.parse_args()
    
    # Process the file
    start_time = time.time()
    process_file(
        args.input, 
        args.model, 
        args.output, 
        args.src_lang, 
        args.tgt_lang, 
        args.device
    )
    end_time = time.time()
    
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()