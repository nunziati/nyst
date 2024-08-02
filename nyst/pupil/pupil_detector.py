import cv2
import numpy as np

class ThresholdingPupilDetector:
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, frame, window_name):
        # Transform the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a Gaussian blur to the grayscale frame
        blurred_roi = cv2.GaussianBlur(gray_frame, (7, 7), 0) # 7x7: dimension of the kernel, 0: "0" indica che la deviazione standard sarà calcolata automaticamente in base alle dimensioni del kernel

        # Return the pupil/iris mask
        _, threshold = cv2.threshold(blurred_roi, self.threshold, 255, cv2.THRESH_BINARY_INV)
        # cv2.imshow(window_name, threshold)

        # Find the contours of the pupil/iris mask
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Control the number of contours found
        if len(contours) == 0:
            return (None, None)
        # Sort the contours in descending order of their area
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # Sort in descending order of the list of contours by their area
        
        # Find the box that contain the contours with the largest area
        (x_min, y_min, w, h) = cv2.boundingRect(contours[0])

        return np.array([x_min + int(w/2), y_min + int(h/2)], dtype=np.int32) # Return the center of the pupil/iris
    
    '''
     # Calculate moments
        M = cv2.moments(contours[0])
        # Calculate centroid
        centroid_x = int(M['m10'] / M['m00'])
        centroid_y = int(M['m01'] / M['m00'])


        return np.array([centroid_x, centroid_y], dtype=np.int32) # Return the center of the pupil/iris'''
    
    def relative_to_absolute(self, relative_position, roi):
        # If the relative position coordinates are None, return None for both coordinates
        if relative_position[0] is None or relative_position[1] is None:
            return None, None
        # Convert relative position to absolute position based on the ROI
        return roi[0] + relative_position[0], roi[1] + relative_position[1]