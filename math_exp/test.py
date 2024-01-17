import os

# Base directory containing the folders
base_dir = './'

# Loop through each folder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    
    # Check if it is a directory
    if os.path.isdir(folder_path):
        # Loop through each file in the folder
        for file in os.listdir(folder_path):
            # Construct the old and new file paths
            old_file_path = os.path.join(folder_path, file)
            new_file_path = os.path.join(base_dir, f"{folder}_{file}")
            
            # Rename the file
            os.rename(old_file_path, new_file_path)
        
        # Optionally, delete the now empty folder
        os.rmdir(folder_path)