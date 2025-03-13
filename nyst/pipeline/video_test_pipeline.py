import cv2
import numpy as np
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader

# Add the 'code' directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.pipeline.first_pipeline import FirstPipeline
from nyst.classifier import SignalNormalizer
from nyst.classifier import NystClassifier


class VideoTestPipeline(FirstPipeline):
    def __init__(self, weights_path, window_size_s, overlapping_s, normalization_std, batch_size=32, device='cpu'):
        super().__init__()

        self.classifier = NystClassifier()
        self.classifier.load_weights(weights_path)

        self.window_size_s = window_size_s
        self.overlapping_s = overlapping_s
        self.window_size_frames = self.window_size_s * self.target_fps
        self.overlapping_frames = self.overlapping_s * self.target_fps

        self.normalizer = SignalNormalizer(normalization_std)

        self.batch_size = batch_size
        self.device = device


    def extract_windows(self, signals, fps):
        windows = []

        for i in range(0, len(signals[0]) - self.window_size_frames, self.window_size_frames - self.overlapping_frames):
            window = signals[:, i:i + self.window_size_frames]
            windows.append(window)

        return torch.from_numpy(np.array(windows))


    def run(self, video_path, output_video):
        # Open the video file   
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (FPS) and resolution of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()

        signals = self.run(video_path, "", 0, False, False)

        speed_resolution = min(signals['speed'].keys())

        signals_array = np.array(
            signals['position']['left'],
            signals['position']['right'],
            signals['speed'][speed_resolution]['left'],
            signals['speed'][speed_resolution]['right']
        )

        signals_array = self.normalizer.normalize(signals_array)

        windows = self.extract_windows(signals_array, fps)

        dataset = TensorDataset(windows)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.classifier.eval()
        self.classifier.to(self.device)

        predictions = []

        with torch.no_grad():
            for x in dataloader:
                x = x.to(self.device)
                output = self.classifier(x)
                predictions.append(output)

        original_overlapping_frames = self.overlapping_s * fps
        predictions = torch.cat(predictions).cpu().numpy()

        
        prediction_per_frame = np.zeros(total_frames)

        # half_overlapping_frames = np.ceil(self.overlapping_frames / 2)

        # prediction_per_frame[:self.window_size_frames - half_overlapping_frames] = predictions[0]

        # for i, prediction in enumerate(predictions[1:], 1):
        #     start_frame = i * (self.window_size_frames - self.overlapping_frames) + half_overlapping_frames
        #     end_frame = start_frame + self.window_size_frames - half_overlapping_frames

        #     prediction_per_frame[start_frame:end_frame] = prediction
        
        for i, prediction in enumerate(predictions):
            window_size_original_frames = self.window_size_frames / self.target_fps * fps
            start_frame = i * (window_size_original_frames - original_overlapping_frames)
            end_frame = start_frame + window_size_original_frames

            prediction_per_frame[start_frame:end_frame] = np.logical_or(prediction_per_frame[start_frame:end_frame], prediction)

        # Create a new video file with the predictions
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter(output_video, fourcc, fps, resolution)

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            if prediction_per_frame[i]:
                # Yolo inference
                result = self.eye_roi_detector.model.predict(frame)
                annotated_frame = result.plot()

                writer.write(annotated_frame)
            else:
                writer.write(frame)

        cap.release()
        writer.release()

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