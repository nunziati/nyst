import cv2
import keras
import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Specifies the second GPU

from nyst.seg_eyes.deeplab_mdl_def import DynamicUpsample
from nyst.seg_eyes.utils import plot_predictions, infer

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
        
        # Create a masked version of the eye frame
        eye_frame_masked = eye_frame.copy()
        eye_frame_masked[prediction_mask == 0] = np.array( [255,255,255])
        
        # Resize the masked frame back to the original size
        eye_frame_masked = cv2.resize(eye_frame_masked, original_size)

        # Optional: Print the segmented eye
        if print_eye:
            plot_predictions(eye_frame, self.COLORMAP, self.model)

        return eye_frame_masked
    


class SegmenterThreshold:
    '''
    Class that performs segmentation of the eye/iris/pupil region in a frame using a deep learning model.

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
        self.label = {
        "background":0,
        "pupil":1,
        "eyes":2,
        "iris":3
        }

        self.model = keras.models.load_model(model_name, custom_objects={'DynamicUpsample': DynamicUpsample})
        self.COLORMAP  = {
            "background": (0, 0, 0),
            "pupil": (255, 0, 0),
            "eyes": (0, 255, 0),
            "iris": (0, 0, 255), 
        }
        # Convert color values from BGR to RGB
        self.COLORMAP = {key: [color[2], color[1], color[0]] for key, color in self.COLORMAP.items()}

    def apply(self, frame, print_eye:bool=False, width:int=448, height:int=448):
        """
        Applies the segmentation model to the given frame to isolate the eye/iris/pupil regions.

        Arguments:
        - frame: The input image frame to be processed.
        - print_eye (bool): Boolean flag indicating whether to print the segmented eye (default: False).
        - width (int): The width to resize the frame for model input (default: 448).
        - height (int): The height to resize the frame for model input (default: 448).

        Returns:
        - A dictionary masked version of the original frame where the eye region is segmented.
        """


        # Resize the frame to the input size of the model
        original_size = (frame.shape[1], frame.shape[0])
        eye_frame = cv2.resize(frame, (width, height))

        # Convert the frame to RGB
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB)

        # Predict the segmentation mask
        image_tensor = np.array(eye_frame)
        prediction_mask = infer(self.model, image_tensor)  # Assuming output shape is (height, width)

        # Create a masked version of the eye frame
        eye_frame_masked = np.zeros_like(eye_frame)

        # Apply the color map to the mask
        for class_name, color in self.COLORMAP.items():
            class_id = self.label[class_name]  # Assuming class IDs are consecutive integers
            class_mask = (prediction_mask == class_id)
            eye_frame_masked[class_mask] = np.array(color)

        # Resize the masked frame back to the original size
        color_mask = cv2.resize(eye_frame_masked, original_size, interpolation=cv2.INTER_NEAREST)
        
        # Resize the prediction mask back to the original size
        prediction_mask = cv2.resize(prediction_mask, original_size, interpolation=cv2.INTER_NEAREST)  

        return prediction_mask, color_mask
    
    
    def apply_segmentation(self, frame, mask, pos, alpha=0.3):
        """
        Applica la maschera con trasparenza al frame.
        
        Parameters:
        - frame: Il frame (immagine) su cui applicare la maschera.
        - mask: La maschera binaria (0 e 255).
        - alpha: Il livello di trasparenza (0.0 = completamente trasparente, 1.0 = completamente opaco).
        """
        # Assicurati che la maschera e il frame abbiano la stessa dimensione
        if frame.shape[:2] != mask.shape:
            raise ValueError("La dimensione della maschera non corrisponde a quella del frame")
        
        # Converti la maschera in formato 8-bit unsigned integer (se non è già in questo formato)
        mask = mask.astype(np.uint8)
        
        # Converti la maschera in formato 3 canali (RGB) per facilitare la fusione
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Aggiungi trasparenza alla maschera: la maschera diventa semi-trasparente
        masked_frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)

        # Mostra il risultato
        cv2.imshow(f"Segmented Frame {pos} eye", masked_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


                