import os
import argparse

import pandas as pd
import numpy as np

from nyst.pipeline import FirstPipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Extract trajectories')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output file name')

    return parser.parse_args()

def main(args = None):
    if args is None:
        args = parse_args()

    input_dir = args['input_dir']
    output = args['output']

    if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")
    
    labels_path = os.path.join(input_dir, 'labels.csv')
    labels_df = pd.read_csv(labels_path, header=0)

    pipeline = FirstPipeline()

    first = ['left', 'right']
    second = ['position', 'speed3', 'speed5', 'speed7', 'speed9']
    third = ['x', 'y']

    signals = [f'{f}_{s}_{t}' for f in first for s in second for t in third]
    patient_list = []
    sample_list = []
    label_list = []

    for idx, row in labels_df.iterrows():
        relative_video_path = row['video']
        video_path = os.path.join(input_dir, relative_video_path)

        video_name = os.path.basename(video_path)

        print("Processing video", video_name)

        patient = int(video_name[:3])
        label = int(row['label'])

        video_path = os.path.join(input_dir, video_name)

        output = pipeline.run(video_path)

        sample = np.concatenate([
            output['position']['left'],
            output['position']['right'],
            output['speed'][3]['left'],
            output['speed'][3]['right'],
            output['speed'][5]['left'],
            output['speed'][5]['right'],
            output['speed'][7]['left'],
            output['speed'][7]['right'],
            output['speed'][9]['left'],
            output['speed'][9]['right']
        ])

        patient_list.append(patient)
        sample_list.append(sample)
        label_list.append(label)

    patients = np.array(patient_list).reshape(-1, 1)
    samples = np.array(sample_list)
    labels = np.array(label_list).reshape(-1, 1)

    data = {
        'signals': signals,
        'patients': patients,
        'samples': samples,
        'labels': labels
    }

    np.save(output, data)


if __name__ == '__main__':
    args = {
        'input_dir': 'data/test_script',
        'output': 'output.csv'
    }
    main(args)