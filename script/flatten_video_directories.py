import argparse
import os
import shutil

import pandas as pd

video_extensions = ['.mp4', '.mkv']

def parse_args():
    parser = argparse.ArgumentParser(description='Flatten videos stored in a directory hierarchy, to a single directory')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')

    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_video_dir = os.path.join(output_dir, 'videos')

    label_file = os.path.join(input_dir, 'labels.csv')
    labels = pd.read_csv(label_file, header=0)

    os.makedirs(output_video_dir, exist_ok=True)

    # Iterate over the rows of the labels dataframe
    for idx, row in labels.iterrows():
        # Extract the patient number from the video name
        patient = row['video'].split('_')[0]

        # Extract the video name from the row
        video = row['video']

        # Construct the new name of the video file
        new_video_name = f"{patient}_{video.split('.')[0].rjust(3, '0')}.mp4"
        new_relative_video_path = os.path.join('videos', new_video_name)
        new_video_path = os.path.join(output_video_dir, new_video_name)

        # Copy the video to the output directory with the new name
        shutil.copy(os.path.join(input_dir, video), new_video_path)

        # Change the video name in the labels dataframe
        labels.at[idx, 'video'] = new_relative_video_path

    # Save the updated labels dataframe
    output_labels_file = os.path.join(output_dir, 'labels.csv')
    labels.to_csv(output_labels_file, index=False)

if __name__ == '__main__':
    main()
