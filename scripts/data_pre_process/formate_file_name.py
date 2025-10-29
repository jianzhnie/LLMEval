import os
import re


def rename_files_in_directory(directory_path):
    """
    Rename files in the specified directory from part_XX_out.jsonl 
    to infer_PCL-Reasoner-57k_part_XX_bz_8.jsonl
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Pattern to match files with part_XX_out.jsonl format
    pattern = re.compile(r'^(part_\d+)\.jsonl_out\.jsonl$')
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            # Extract the number part (e.g., 00, 01, etc.)
            base_part = match.group(1)
            print(base_part)
            # Create old and new file paths
            old_path = os.path.join(directory_path, filename)
            new_filename = f"infer_PCL-Reasoner-57k_{base_part}_bz8.jsonl"
            print(new_filename)
            new_path = os.path.join(directory_path, new_filename)
            
            # Rename the file
            try:
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except OSError as e:
                print(f"Error renaming {filename}: {e}")


# Main execution
if __name__ == "__main__":
    # Specify the directory containing the files
    directory_path = "/home/jianzhnie/llmtuner/llm/LLMEval/output/PCL-Reasoner-57k"
    
    directory_path = "/home/jianzhnie/llmtuner/llm/LLMEval/data/merged_skywork_R10528_nvidia_57K/"
    # Rename files in the specified directory
    rename_files_in_directory(directory_path)
    


    