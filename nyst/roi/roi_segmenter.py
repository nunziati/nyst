import cv2
import keras
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specifica la seconda GPU

from nyst.seg_eyes.deeplab_mdl_def import DynamicUpsample
from nyst.seg_eyes.utils import plot_predictions, infer, decode_segmentation_masks

# Class that defines a method to calculate the ROI boxes of each eye separately.
class FirstEyeRoiSegmenter:
    def __init__(self, model_name): # You can use backend=yolov8 for face detection.

        self.model = keras.models.load_model(model_name, custom_objects={'DynamicUpsample': DynamicUpsample})
        self.COLORMAP  = {
    "background": [0, 0, 0],  # BGR
    "eyes": [1, 1, 1],  # BGR  
    }
        self.COLORMAP = {key: [color[2], color[1], color[0]] for key, color in self.COLORMAP.items()}
    def apply(self, frame, print_eye=False, width=448, height=448):
        
        # Allarghiamo la regione dell'occhio
        original_size = (frame.shape[1], frame.shape[0])
        eye_frame = cv2.resize(frame, (width, height))

        # Mettiamo in RGB
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)

        # Prediction mask con classe 0 ed 1
        image_tensor = np.array(eye_frame)
        prediction_mask = infer(self.model, image_tensor)
        
        # To do: Eliminazione blob piccoli, verifichiamo area occhio del detetcted box

        # Prediction mask con bianco il background e l'occhio seggmentato in RGB
        eye_frame_masked = eye_frame.copy()
        eye_frame_masked[prediction_mask == 0] = np.array( [255,255,255])
        eye_frame_masked = cv2.resize(eye_frame_masked, original_size) #Riportiamo l'occhio alla dim originale

        if print_eye:
            plot_predictions(eye_frame, self.COLORMAP, self.model)

        return eye_frame_masked

        