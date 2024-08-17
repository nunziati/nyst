import os
import shutil

# List of video file extensions to be considered
video_extensions = ['.mp4', '.mkv']

# Function to flatten videos stored in a directory hierarchy to a single directory
def flattenVideoDirectories(input_dir, output_dir):
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)


    # Loop through each item in the input directory
    for subfolder in os.listdir(input_dir):
        # Get the full path of the subfolder
        subfolder_path = os.path.join(input_dir, subfolder)

        # Skip if it's not a directory
        if not os.path.isdir(subfolder_path):
            continue
        
        # Loop through each item in the subfolder
        for video in os.listdir(subfolder_path):
            # Get the full path of the video file
            video_path = os.path.join(subfolder_path, video)

            # Check if the file is a video
            if not any(video.endswith(ext) for ext in video_extensions):
                continue
            
            # Copy the video file to the output directory with a new name
            shutil.copy(video_path, os.path.join(output_dir, f"{subfolder}_{video}"))

