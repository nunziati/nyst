from deepface import DeepFace
import numpy as np
import cv2

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Imposta il dispositivo GPU desiderato (0, 1, 2, ecc.)


from .roi import FirstRoi

# Class that defines a method to calculate the ROI boxes of each eye separately.
class FirstEyeRoiDetector:
    def __init__(self, backend): # You can use backend=yolov8 for face detection.
        self.backend = backend 

    def apply(self, frame): 
        # Extract first face box using DeepFace Architecture
        face_obj = DeepFace.extract_faces(frame, target_size=frame.shape[1:], detector_backend=self.backend)[0]
        
        # Extract the coordinates of the left eye and right eye (one point for each eye)
        left_eye = np.array(face_obj['facial_area']['left_eye'], dtype=int)
        right_eye = np.array(face_obj['facial_area']['right_eye'], dtype=int)

        # Disegna i punti sull'immagine
        cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)  # -1 per disegnare un cerchio pieno
        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)

        # Mostra l'immagine con i punti
        cv2.imshow("Image with Points", frame)



        # Approximation of the eucledian distanza between the two extreme points of the rectangles
        distance = int(np.linalg.norm(right_eye - left_eye))

        half_distance_w = distance // 4
        half_distance_h = distance // 8
        half_distance = np.array([half_distance_w, half_distance_h])

        # Definition of the left and right eyes ROI
        left_eye_roi = np.concatenate([left_eye - half_distance, left_eye + half_distance])
        right_eye_roi = np.concatenate([right_eye - half_distance, right_eye + half_distance])
        
        return FirstRoi(left_eye_roi, right_eye_roi) 
