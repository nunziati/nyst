import cv2
import numpy as np

class ThresholdingPupilDetector:
    def __init__(self):
        pass

    def apply(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_roi = cv2.GaussianBlur(gray_frame, (7, 7), 0)

        _, threshold = cv2.threshold(blurred_roi, 50, 255, cv2.THRESH_BINARY_INV)

        contours, _, = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        if len(contours) == 0:
            return (None, None)
        
        (x_min, y_min, w, h) = cv2.boundingRect(contours[0])

        return np.array([x_min + int(w/2), y_min + int(h/2)], dtype=np.int32)