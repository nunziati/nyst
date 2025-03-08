import numpy as np
import cv2
import traceback
import os
import sys
from ultralytics import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Imposta il dispositivo GPU desiderato (0, 1, 2, ecc.)

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Class to detect eyes' ROIs using YOLOv11
class FirstEyeRoiDetector:
    """
    Class that defines a method to calculate the ROI (Region of Interest) boxes for each eye separately using YOLOv11.

    Attributes:
    - model_path: Path to the YOLO model weights.
    """
    def __init__(self, model_path):
        # Load the YOLO model
        self.model = YOLO(model_path)  # Ensure model_path points to a YOLOv11 weights file

    def apply(self, frame, count_from_lastRoiupd, old_left_eye, old_right_eye):
        """
        Applies YOLO detection and extracts the ROIs for the eyes.

        Arguments:
        - frame: The current video frame to process.

        Returns:
        - A tuple containing the ROI boxes for the left and right eyes.
        """
        try:
            # Perform detection
            results = self.model.predict(frame, conf=0.5) # Adjust confidence threshold as needed

            # Extract detections
            detections = results[0].boxes.data.cpu().numpy()  # Assuming YOLO outputs boxes in (x1, y1, x2, y2) format
            eye_boxes = []
            
            # Add null boxes detection
            if len(detections) < 2:
                for _ in range(1-len(detections)):
                    eye_boxes.append(np.zeros(4))

            # Add correct boxes detected
            for det in detections:
                # Assuming 'eye' is a specific class
                class_id = int(det[5])  # The class ID (usually the last element in the detection array)
                if class_id == 0:  # Assuming 'eye' is class 0
                    eye_boxes.append(det[:4])  # Append the bounding box (x1, y1, x2, y2)
            
            # Control all the boxes and adjest the boxes value
            if len(eye_boxes) < 3:
                # Sort boxes by their horizontal position (x-coordinate)
                eye_boxes = sorted(eye_boxes, key=lambda box: box[0])

                # Extract boxes of left and right eyes
                left_eye, right_eye = eye_boxes[:2]  
            
                # Adjust boxes of the left and right eyes
                if np.array_equal(left_eye, np.zeros(4)) and np.array_equal(right_eye, np.zeros(4)): # Both eyes are not detected
                    # Adjust roi value
                    left_eye = old_left_eye
                    right_eye = old_right_eye 
                    # Increment of the counter
                    count_from_lastRoiupd += 1
                
                elif np.array_equal(left_eye, np.zeros(4)): # Left eye is not detected
                    # Adjust roi value
                    left_eye = old_left_eye
                    old_right_eye = right_eye
                    # Increment of the counter
                    count_from_lastRoiupd += 1

                elif np.array_equal(right_eye, np.zeros(4)): # Right eye is not detected
                    # Adjust roi value
                    right_eye = old_right_eye
                    old_left_eye = left_eye
                    # Increment of the counter
                    count_from_lastRoiupd += 1
                
                else: # Both eyes are detected
                    old_left_eye = left_eye
                    old_right_eye = right_eye
                    # Counter reset
                    count_from_lastRoiupd = 0
                
                return left_eye, right_eye, old_left_eye, old_right_eye, count_from_lastRoiupd

            else:
                raise ValueError("Too much eyes in the video, please check your video!")   

        except Exception as e:
            print(f"An error occurred: {e}")
            traceback.print_exc()
            


if __name__ == "__main__":

    # Path to the YOLO model weights
    model_path = 'D:/model_yolo/best_yolo11m.pt'  # Update with actual path
    detector = FirstEyeRoiDetector(model_path)

    # Percorso del video
    video_path = "D:/prova/002_001.mp4"

    # Apri il video
    cap = cv2.VideoCapture(video_path)

    # Salvataggio del video processato (opzionale)
    output_path = "D:/video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Fine del video o errore nella lettura del frame.")
            break

        # Applica il rilevamento delle ROI
        left_eye_box, right_eye_box = detector.apply(frame)

        # Disegna le bounding box sugli occhi, se rilevate
        if left_eye_box is not None:
            cv2.rectangle(frame, 
                        (int(left_eye_box[0]), int(left_eye_box[1])), 
                        (int(left_eye_box[2]), int(left_eye_box[3])), 
                        (255, 0, 0), 2)  # Blu per l'occhio sinistro

        if right_eye_box is not None:
            cv2.rectangle(frame, 
                        (int(right_eye_box[0]), int(right_eye_box[1])), 
                        (int(right_eye_box[2]), int(right_eye_box[3])), 
                        (0, 255, 0), 2)  # Verde per l'occhio destro

        # Mostra il frame processato
        cv2.imshow("Detected Eyes", frame)

        # Scrivi il frame processato nel video di output
        out.write(frame)

        # Premi 'q' per uscire
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia risorse
    cap.release()
    out.release()
    cv2.destroyAllWindows()

