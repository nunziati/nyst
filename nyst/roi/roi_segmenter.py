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
    '''
    Class that performs segmentation of the eye region in a frame using a deep learning model.

    Attributes:
    - model: The deep learning model used for segmentation.
    - COLORMAP: A dictionary that maps class names to BGR color values.
    '''
    def __init__(self, model_name:str):
        '''
        Initializes the FirstEyeRoiSegmenter with the specified model.

        Arguments:
        - model_name: The filename of the pre-trained model to be loaded.
        '''
        self.model = keras.models.load_model(model_name, custom_objects={'DynamicUpsample': DynamicUpsample})
        self.COLORMAP  = {
    "background": [0, 0, 0],  # BGR for background
    "eyes": [1, 1, 1],  # BGR for eyes
    }
        # Convert color values from BGR to RGB
        self.COLORMAP = {key: [color[2], color[1], color[0]] for key, color in self.COLORMAP.items()}
    
    def apply(self, frame, print_eye:bool=False, width:int=448, height:int=448):
        """
        Applies the segmentation model to the given frame to isolate the eye region.

        Arguments:
        - frame: The input image frame to be processed.
        - print_eye (bool): Boolean flag indicating whether to print the segmented eye (default: False).
        - width (int): The width to resize the frame for model input (default: 448).
        - height (int): The height to resize the frame for model input (default: 448).

        Returns:
        - A masked version of the original frame where the eye region is segmented.
        """
        
        # Resize the frame to the input size of the model
        original_size = (frame.shape[1], frame.shape[0])
        eye_frame = cv2.resize(frame, (width, height))

        # Convert the frame to RGB
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)

        # Predict the segmentation mask
        image_tensor = np.array(eye_frame)
        prediction_mask = infer(self.model, image_tensor)
        
        # To do: Eliminazione blob piccoli, verifichiamo area occhio del detetcted box

        # Create a masked version of the eye frame
        eye_frame_masked = eye_frame.copy()
        eye_frame_masked[prediction_mask == 0] = np.array( [255,255,255])
        
        # Resize the masked frame back to the original size
        eye_frame_masked = cv2.resize(eye_frame_masked, original_size)

        # Optional: Print the segmented eye
        if print_eye:
            plot_predictions(eye_frame, self.COLORMAP, self.model)

        return eye_frame_masked

        