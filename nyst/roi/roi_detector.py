import cv2
from deepface import DeepFace

from .roi import FirstRoi

class FirstEyeRoiDetector:
    def __init__(self, backend):
        self.backend = backend

    def apply(self, frame):
        face_obj = DeepFace.extract_faces(frame, target_size=frame.shape[1:], detector_backend=self.backend)[0]
        return FirstRoi(face_obj)
