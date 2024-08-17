import cv2
import numpy as np

class ThresholdingPupilDetector:
    '''
    Class that detects the pupil or iris in a given ROI frame using thresholding.

    Attributes:
    - threshold: The threshold value used for binary thresholding of the image.
    '''
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, frame):
        '''
        Detects the pupil or iris in the given frame using thresholding and contour analysis.

        Arguments:
        - frame: The input image frame in which the pupil/iris is to be detected.

        Returns:
        - A numpy array containing the coordinates of the center of the detected pupil/iris, 
          or (None, None) if no contours are found.
        '''
        # Transform the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply a Gaussian blur to the grayscale frame
        blurred_roi = cv2.GaussianBlur(gray_frame, (7, 7), 0) # 7x7: dimension of the kernel, 0: "0" indicates that the standard deviation will be automatically calculated based on the size of the kernel

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
    
    def relative_to_absolute(self, relative_position:np.array, roi):
        '''
        Converts the relative coordinates of a position within a region of interest (ROI) 
        to absolute coordinates in the original image.

        Arguments:
        - relative_position: An array (x, y) coordinates of the positions.
        - roi: (x, y) coordinates of the top-left corner of the ROI 
            in the original image.

        Returns:
        - A tuple containing the absolute (x, y) coordinates of the position in the original image.
        If either coordinate in relative_position is None, returns (None, None).
        '''
        # If the relative position coordinates are None, return None for both coordinates
        if relative_position[0] is None or relative_position[1] is None:
            return None, None
        # Convert relative position to absolute position based on the ROI
        return roi[0] + relative_position[0], roi[1] + relative_position[1]