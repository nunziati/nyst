from deepface import DeepFace
import numpy as np


from .roi import FirstRoi

# Class that defines a method to calculate the ROI boxes of each eye separately.
class FirstEyeRoiDetector:
    def __init__(self, backend): # You can use backend=yolov8 for face detection.
        self.backend = backend 

    def apply(self, frame): 
        # Extract first face box using DeepFace Architecture
        face_obj = DeepFace.extract_faces(frame, target_size=frame.shape[1:], detector_backend=self.backend)[0]
        
        # Extract the coordinates of the left eye and right eye (one point for each eye)
        left_eye = np.array(face_obj['face']['facial_area']['left_eye'], dtype=int)
        right_eye = np.array(face_obj['face']['facial_area']['right_eye'], dtype=int)

        # Approximation of the eucledian distanza between the two extreme points of the rectangles
        distance = int(np.linalg.norm(right_eye - left_eye))

        half_distance = distance // 2

        # Definition of the left and right eyes ROI
        left_eye_roi = np.concatenate([left_eye - half_distance, left_eye + half_distance])
        right_eye_roi = np.concatenate([right_eye - half_distance, right_eye + half_distance])
        
        return FirstRoi(left_eye_roi, right_eye_roi) 
