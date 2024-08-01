import argparse
import os
import shutil

# List of video file extensions to be considered
video_extensions = ['.mp4', '.mkv']

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Flatten videos stored in a directory hierarchy, to a single directory')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    return parser.parse_args()

# Function to flatten videos stored in a directory hierarchy to a single directory
def main():
    # Parse the arguments
    args = parse_args()

    input_dir = args.input_dir # Get input directory
    output_dir = args.output_dir # Get output directory 

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

if __name__ == '__main__':
    main()
