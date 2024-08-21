import cv2
import numpy as np
import sys
import keras

sys.path.append(r'C:/Users/andre/OneDrive\Desktop/MAGISTRALE/Tesi/nyst')
from nyst.seg_eyes.deeplab_mdl_def import DeeplabV3Plus
from nyst.seg_eyes.deeplab_mdl_def import DynamicUpsample
from nyst.seg_eyes.utils import infer, decode_segmentation_masks, get_overlay, plot_samples_matplotlib

def plot_predictions(frame, colormap, model):
    image_tensor = np.array(frame)
    prediction_mask = infer(model, image_tensor)
    prediction_colormap = decode_segmentation_masks(prediction_mask, colormap, n_classes=2)
    overlay = get_overlay(image_tensor, prediction_colormap)
    plot_samples_matplotlib([image_tensor, prediction_colormap, overlay], figsize=(18,14))
    # Chiudi la finestra dopo 1 millisecondo
    cv2.waitKey(0)

if __name__ == '__main__':

    # Inizializza il lettore video
    video_path = 'C:/Users/andre/Downloads/Telegram Desktop/1.mp4'
    # Inizializza il lettore video
    cap = cv2.VideoCapture(video_path)

    # Controlla se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore nell'apertura del video")
        exit()    

    # Definisco il modello
    model = DeeplabV3Plus(num_classes=2)

    # Carica i pesi preaddestrati
    model = keras.models.load_model('C:/Users/andre/OneDrive/Desktop/MAGISTRALE/Tesi/nyst/nyst/seg_eyes/model.h5', custom_objects={'DynamicUpsample': DynamicUpsample})

    COLORMAP = patch_colors_bgr_01 = {
    "background": [0, 0, 0],  # BGR
    "eyes": [1, 1, 1],  # BGR  
    }
    COLORMAP = {key: [color[2], color[1], color[0]] for key, color in COLORMAP.items()}
    