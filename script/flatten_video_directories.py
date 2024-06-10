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

    os.makedirs(output_dir, exist_ok=True)

    for subfolder in os.listdir(input_dir):
        subfolder_path = os.path.join(input_dir, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        for video in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video)

            if not any(video.endswith(ext) for ext in video_extensions):
                continue

            shutil.copy(video_path, os.path.join(output_dir, f"{subfolder}_{video}"))

if __name__ == '__main__':
    main()
