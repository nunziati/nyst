import cv2
import numpy as np
import os
import sys
import plotly.graph_objects as go
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# Add the 'code' directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.pipeline import FirstPipeline
from nyst.classifier import NystClassifier
from nyst.dataset import NystInferenceDataset

class VideoTestPipeline(FirstPipeline):
    def __init__(self, classifier_weights, roi_detector_weights, eye_segmenter_weights, logfile_path, window_size_s, overlapping_s, normalization_std, batch_size=32, device='cpu'):
        super().__init__(roi_detector_weights, eye_segmenter_weights, logfile_path, interpolate=None)

        self.classifier = NystClassifier(nf=16)
        self.classifier.load_weights(classifier_weights)

        self.window_size_s = window_size_s
        self.overlapping_s = overlapping_s
        self.window_size_frames = self.window_size_s * self.target_fps
        self.overlapping_frames = self.overlapping_s * self.target_fps

        self.normalization_std = normalization_std

        self.batch_size = batch_size
        self.device = device


    def extract_windows(self, signals):
        windows = []

        for i in range(0, len(signals[0]) - self.window_size_frames, self.window_size_frames - self.overlapping_frames):
            window = signals[:, i:i + self.window_size_frames]
            windows.append(window)

        return np.array(windows)


    def run(self, video_path, output_video, output_plot):
        # Open the video file   
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (FPS) and resolution of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        print("Extracting signals...")
        signals = super().run(video_path, "", 0, False, False)
        print("Signals extracted")

        speed_resolution = min(signals['speed'].keys())

        signals_array = np.array([
            signals['position']['left'].T,
            signals['position']['right'].T,
            signals['speed'][speed_resolution]['left'].T,
            signals['speed'][speed_resolution]['right'].T
        ]).reshape(8, -1)

        print("Extracting windows...")
        windows = self.extract_windows(signals_array)
        print("Windows extracted")


        dataset = NystInferenceDataset(windows, self.normalization_std)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.classifier.eval()
        self.classifier.to(self.device)

        predictions = []

        print("Running inference...")
        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                output = self.classifier(x)
                predictions.append(output)

        original_overlapping_frames = self.overlapping_s * fps
        
        predictions = torch.cat(predictions).cpu().numpy()

        print(predictions)

        predictions = predictions > 0.8


        print("Inference done")
        prediction_per_frame = np.zeros(total_frames)

        # half_overlapping_frames = np.ceil(self.overlapping_frames / 2)

        # prediction_per_frame[:self.window_size_frames - half_overlapping_frames] = predictions[0]

        # for i, prediction in enumerate(predictions[1:], 1):
        #     start_frame = i * (self.window_size_frames - self.overlapping_frames) + half_overlapping_frames
        #     end_frame = start_frame + self.window_size_frames - half_overlapping_frames

        #     prediction_per_frame[start_frame:end_frame] = prediction
        
        for i, prediction in enumerate(predictions):
            window_size_original_frames = self.window_size_frames / self.target_fps * fps
            start_frame = int(i * (window_size_original_frames - original_overlapping_frames))
            end_frame = int(start_frame + window_size_original_frames)

            prediction_per_frame[start_frame:end_frame] = np.logical_or(prediction_per_frame[start_frame:end_frame], prediction)

        # Create a new video file with the predictions
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(output_video, fourcc, fps, resolution)

        print("Creating annotated video...")
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            if prediction_per_frame[i]:
                # Yolo inference
                result = self.eye_roi_detector.model.predict(frame, verbose=False)
                annotated_frame = result[0].plot()

                writer.write(annotated_frame)
            else:
                writer.write(frame)

        cap.release()
        writer.release()

        print("Annotated video created")

        print("Creating plot...")

        zero_mean_signals_array = signals_array - np.mean(signals_array, axis=1).reshape(8, 1)
        names = ['Position Left X', 'Position Left Y', 'Position Right X', 'Position Right Y', 'Speed Left X', 'Speed Left Y', 'Speed Right X', 'Speed Right Y']
        # Generate plot with Plotly
        fig = go.Figure()

        # Add traces for the 4 signals
        for name, signal in zip(names, zero_mean_signals_array):
            fig.add_trace(go.Scatter(y=signal, mode='lines', name=name))

        """# Highlight parts with label=1
        label_indices = np.where(prediction_per_frame == 1)[0]

        # Find rectangle extremes
        start_indices = []
        end_indices = []

        start_indices.append(label_indices[0])
        for i in range(1, len(label_indices)):
            if label_indices[i] != label_indices[i - 1] + 1:
                end_indices.append(label_indices[i - 1] + 1)
                start_indices.append(label_indices[i])
        end_indices.append(label_indices[-1] + 1)

        for start_idx, end_idx in zip(start_indices, end_indices):
            fig.add_vrect(x0=start_idx, x1=end_idx, fillcolor="LightSalmon", opacity=0.5, line_width=0)"""

        # Highlight parts with prediction=1
        prediction_indices = np.where(prediction_per_frame == 1)[0]

        # Find rectangle extremes
        start_indices = []
        end_indices = []

        start_indices.append(prediction_indices[0])
        for i in range(1, len(prediction_indices)):
            if prediction_indices[i] != prediction_indices[i - 1] + 1:
                end_indices.append(prediction_indices[i - 1] + 1)
                start_indices.append(prediction_indices[i])
        end_indices.append(prediction_indices[-1] + 1)

        for start_idx, end_idx in zip(start_indices, end_indices):
            fig.add_vrect(x0=start_idx, x1=end_idx, fillcolor="LightGreen", opacity=0.2, line_width=0)

        # Save the plot
        fig.write_html(output_plot)
        print("Plot created")

    def run_batch(self, video_folder, output_folder):
        for root, _, files in os.walk(video_folder):
            for file in files:
                if file.endswith(('.mp4', '.avi', '.mov')):
                    input_video_path = os.path.join(root, file)
                    relative_path = os.path.relpath(root, video_folder)
                    output_dir = os.path.join(output_folder, relative_path)
                    os.makedirs(output_dir, exist_ok=True)
                    output_video_path = os.path.join(output_dir, file)
                    self.run(input_video_path, output_video_path)

if __name__ == '__main__':
    weights_path = '/repo/porri/nyst/models_10/run-20250324_160114-gihkxu9t/best_model.pth'
    roi_detector_weights = '/repo/porri/nyst/yolo_models/best_yolo11m.pt'
    eye_segmenter_weights = '/repo/porri/eyes_seg_threshold.h5'
    logfile_path = '/repo/porri/output_infer_video/logfile.txt'
    window_size_s = 5
    overlapping_s = 3
    normalization_std = np.load('/repo/porri/nyst_labelled_videos/std.npy')

    input_video = '/repo/porri/prova/042_001.mp4'
    output_video = '/repo/porri/output_infer_video/042_001.mp4'
    output_plot = '/repo/porri/output_infer_video/042_001.html'

    pipeline = VideoTestPipeline(weights_path, roi_detector_weights, eye_segmenter_weights, logfile_path, window_size_s, overlapping_s, normalization_std)
    pipeline.run(input_video, output_video, output_plot)