import os
import shutil

def process_directory(directory_path):
    # Get all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Filter for CSV files
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Group files by their prefix
    file_groups = {}
    for file in csv_files:
        if '_' in file:
            parts = file.rsplit('_', 1)
            if len(parts) == 2 and parts[1].startswith('tag'):
                prefix = parts[0]
                tag = parts[1][3:-4]  # Remove 'tag' and '.csv'
            else:
                prefix = file[:-4]  # Remove '.csv'
                tag = None
        else:
            prefix = file[:-4]  # Remove '.csv'
            tag = None
        
        if prefix not in file_groups:
            file_groups[prefix] = {'base': None, 'tag': None}
        
        if tag is None:
            file_groups[prefix]['base'] = file
        else:
            file_groups[prefix]['tag'] = file

    # Process each group
    for prefix, group in file_groups.items():
        if group['base']:
            source_path = os.path.join(directory_path, group['base'])
            destination_path = os.path.join(directory_path, "OG.csv")
            shutil.copy2(source_path, destination_path)
            print(f"Processed {group['base']} -> OG.csv")
        
        if group['tag']:
            source_path = os.path.join(directory_path, group['tag'])
            tag = group['tag'].rsplit('_', 1)[1][3:-4]  # Extract tag
            destination_path = os.path.join(directory_path, f"{tag}.csv")
            shutil.copy2(source_path, destination_path)
            print(f"Processed {group['tag']} -> {tag}.csv")
        
        if not group['base'] and not group['tag']:
            print(f"Skipping group {prefix} as it doesn't match the expected pattern.")

# Example usage
directory_path = "./emotibit_parsed"
process_directory(directory_path)