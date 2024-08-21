import cv2
import numpy as np
import sys
import keras
from deepface import DeepFace


sys.path.append(r'C:/Users/andre/OneDrive\Desktop/MAGISTRALE/Tesi/nyst')
from nyst.seg_eyes.deeplab_mdl_def import DeeplabV3Plus
from nyst.seg_eyes.deeplab_mdl_def import DynamicUpsample
from nyst.seg_eyes.utils import infer, decode_segmentation_masks, get_overlay, plot_samples_matplotlib


def apply_ROI_select(frame,backend="yolov8"):
                
        # Extract first face box using DeepFace Architecture
        face_obj = DeepFace.extract_faces(frame, target_size=frame.shape[1:], detector_backend=backend)[0]

        
        # Extract the coordinates of the left eye and right eye (one point for each eye)
        left_eye = np.array(face_obj['facial_area']['left_eye'], dtype=int)
        right_eye = np.array(face_obj['facial_area']['right_eye'], dtype=int)

        # Approximation of the eucledian distanza between the two extreme points of the rectangles
        distance = int(np.linalg.norm(right_eye - left_eye))

        half_distance_w = distance // 4
        half_distance_h = distance // 8
        half_distance = np.array([half_distance_w, half_distance_h])
        

        # Definition of the left and right eyes ROI
        left_eye_roi = np.concatenate([left_eye - half_distance, left_eye + half_distance])
        right_eye_roi = np.concatenate([right_eye - half_distance, right_eye + half_distance])
        
        return left_eye_roi, right_eye_roi

def apply_crop(frame, roi): # Crop the frame with the specified ROI
        return frame[roi[1]:roi[3], roi[0]:roi[2]]

def ROI_sep_eye(frame, update_roi= True):
        # Uptdating eyes ROI
        if update_roi:
            # Calculate the ROI of left and right eyes
            left_eye_roi, right_eye_roi = apply_ROI_select(frame) 

        # Apply ROI to the selected frame and assign the result to specific variables
        left_eye_frame = apply_crop(frame, left_eye_roi)
        right_eye_frame = apply_crop(frame, right_eye_roi)
        
        return left_eye_frame, right_eye_frame

def plot_predictions(frame, colormap, model):
    image_tensor = np.array(frame)
    prediction_mask = infer(model, image_tensor)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=2)
    overlay = get_overlay(image_tensor, prediction_colormap)
    plot_samples_matplotlib([image_tensor, prediction_colormap, overlay], figsize=(18,14))
    # Chiudi la finestra dopo 1 millisecondo
    cv2.waitKey(1)



if __name__ == '__main__':

    # Inizializza il lettore video
    video_path = 'C:/Users/andre/Downloads/Telegram Desktop/2.mp4'
    # Inizializza il lettore video
    cap = cv2.VideoCapture(video_path)

    # Controlla se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        exit()    

    # Definisco il modello
    #model = DeeplabV3Plus(num_classes=2)

    # Carica i pesi preaddestrati
    model = keras.models.load_model('C:/Users/andre/OneDrive/Desktop/MAGISTRALE/Tesi/nyst/nyst/seg_eyes/model.h5', custom_objects={'DynamicUpsample': DynamicUpsample})

    # Crea finestre di visualizzazione con dimensioni iniziali adatte al frame
    '''cv2.namedWindow('Segmented Left Eye', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmented Left Eye')
    cv2.namedWindow('Segmented Right Eye', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Segmented Right Eye', 480, 270)'''

    COLORMAP = patch_colors_bgr_01 = {
    "background": [0, 0, 0],  # BGR
    "eyes": [1, 1, 1],  # BGR  
    }
    COLORMAP = {key: [color[2], color[1], color[0]] for key, color in COLORMAP.items()}
    
    try:
        # Ciclo per ogni frame del video
        while True:
            # Leggi il frame corrente
            ret, frame = cap.read()

            # Controlla se il frame è stato letto correttamente
            if not ret:
                break 

            # Ruota il frame di 90° in senso orario
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            #rotated_frame = frame
            
            
            left_eye_frame, right_eye_frame = ROI_sep_eye(rotated_frame)
            

            left_eye_frame = cv2.resize(left_eye_frame, (448,448))
            right_eye_frame = cv2.resize(right_eye_frame, (448,448))

            left_eye_frame = cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2RGB)
            right_eye_frame = cv2.cvtColor(right_eye_frame, cv2.COLOR_BGR2RGB)

            #pred = model.predict(np.expand_dims((frame_tensor), axis=0), verbose=0)
            plot_predictions(left_eye_frame, COLORMAP, model)
            plot_predictions(right_eye_frame, COLORMAP, model)


            # Mostra i risultati grafici
            #cv2.imshow('Segmented Left Eye', bw_image_normalized)
            #cv2.imshow('Segmented Right Eye', rotated_frame)

            # Esce se viene premuto 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Rilascia le risorse
        cv2.destroyAllWindows()

