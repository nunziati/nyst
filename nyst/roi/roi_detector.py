from deepface import DeepFace
import numpy as np
import cv2
from time import sleep
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Imposta il dispositivo GPU desiderato (0, 1, 2, ecc.)

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .roi import FirstRoi

# Class that defines a method to calculate the ROI boxes of each eye separately.
class FirstEyeRoiDetector:
    '''
    Class that defines a method to calculate the ROI (Region of Interest) boxes for each eye separately.

    Attributes:
    - backend: The face detection backend to be used (e.g., yolov8).
    - last_left_eye: Stores the last known position of the left eye for cases when the face is not detected.
    - last_right_eye: Stores the last known position of the right eye for cases when the face is not detected.
    '''
    def __init__(self, backend): # You can use backend=yolov8 for face detection
        self.backend = backend 
        self.last_left_eye = None
        self.last_right_eye = None

    def apply(self, frame, idx_frame:int):
        """
        Applies the face detection and calculates the ROI boxes for each eye in the given frame.

        Arguments:
        - frame: The current video frame to process.
        - idx_frame: The index of the current frame in the video sequence.

        Returns:
        - FirstRoi: An object containing the ROI boxes for the left and right eyes.
        """
        try:
            # Define the target size for the face detection
            target_size = (frame.shape[0], frame.shape[1])
            
            # Extract first face box using DeepFace Architecture
            faces = DeepFace.extract_faces(frame, target_size=target_size, detector_backend=self.backend, enforce_detection=False) # enforce_detection=False to manage no detected face case
            
            if faces: # Detected face Case
                face_obj = faces[0]
                # Extract the coordinates of the left eye and right eye (one point for each eye)
                left_eye = np.array(face_obj['facial_area']['left_eye'], dtype=int)
                right_eye = np.array(face_obj['facial_area']['right_eye'], dtype=int)
                
                if left_eye is not None and right_eye is not None: # Both eyes detected
                    # Update the last known positions
                    self.last_left_eye = left_eye
                    self.last_right_eye = right_eye 
                else:
                    if left_eye is None and right_eye is None: # Both eyes are not detected
                        left_eye = self.last_left_eye
                        right_eye = self.last_left_eye
                    
                    elif left_eye is None: # Left eye is not detected
                        left_eye = self.last_left_eye
                        self.last_right_eye = right_eye

                    else: # Right eye is not detected
                        right_eye = self.last_right_eye
                        self.last_left_eye = left_eye
                       
            else: # Case of no detected face and middle
                print("No faces detected, using last known positions.")
                '''
                if idx_frame == 0: # First frame without detected faces Case
                    self.last_left_eye = None
                    self.last_right_eye = None
                    return FirstRoi(None, None)
                
                elif self.last_left_eye is not None and self.last_right_eye is not None: # Previous position detected Case
                    left_eye = self.last_left_eye
                    right_eye = self.last_right_eye
                
                    
                else: # Previous position does not detected Case
                '''
                #print("No previous eye positions available.")
                return FirstRoi(None, None)
        
            # Draw the points on the image
            cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)  # -1 for filled circle
            cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
            
            # Mostra l'immagine con i punti
            #cv2.imshow("Image with Points", frame)
            
            # Approximation of the Euclidean distance between the two extreme points of the rectangles
            distance = int(np.linalg.norm(right_eye - left_eye))
            
            # Distances Augmentation 
            half_distance_w = distance // 4
            half_distance_h = distance // 6
            half_distance = np.array([half_distance_w, half_distance_h])
            
            # Definition of the left and right eyes ROI
            left_eye_roi = np.concatenate([left_eye - half_distance, left_eye + half_distance])
            right_eye_roi = np.concatenate([right_eye - half_distance, right_eye + half_distance])
            
            return FirstRoi(left_eye_roi, right_eye_roi)
        
        except Exception as e:
            print(f"An error occurred: {e}")
           


        '''# Define the target size for the face detection
        target_size = (frame.shape[0],frame.shape[1])
        # Extract first face box using DeepFace Architecture
        face_obj = DeepFace.extract_faces(frame, target_size=target_size, detector_backend=self.backend, enforce_detection=False)[0] # target_size=frame.shape[1:]

        # Extract the coordinates of the left eye and right eye (one point for each eye)
        left_eye = np.array(face_obj['facial_area']['left_eye'], dtype=int)
        right_eye = np.array(face_obj['facial_area']['right_eye'], dtype=int)
        
        # Disegna i punti sull'immagine
        cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)  # -1 per disegnare un cerchio pieno
        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)

        # Mostra l'immagine con i punti
        #cv2.imshow("Image with Points", frame)

        # Approximation of the eucledian distanza between the two extreme points of the rectangles
        distance = int(np.linalg.norm(right_eye - left_eye))

        half_distance_w = distance // 4
        half_distance_h = distance // 6
        half_distance = np.array([half_distance_w, half_distance_h])

        # Definition of the left and right eyes ROI
        left_eye_roi = np.concatenate([left_eye - half_distance, left_eye + half_distance])
        right_eye_roi = np.concatenate([right_eye - half_distance, right_eye + half_distance])
        
        return FirstRoi(left_eye_roi, right_eye_roi) '''
        
        
