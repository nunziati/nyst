import cv2
import numpy as np
import csv

class CenterPupilIrisRegionDetector:
    '''
    Class that detects the pupil or iris in a given ROI frame using thresholding.

    Attributes:
    - threshold: The threshold value used for binary thresholding of the image.
    '''
    def __init__(self, threshold):
        #self.save_threshold_interval_counts = {"left_pupil+iris":0,"left_pupil":0,"left_pupil_list":[],"left_iris_list":[],"right_pupil+iris":0,"right_pupil":0,"right_pupil_list":[],"right_iris_list":[]}
        self.threshold = threshold
        
    def apply(self, frame, mask, count, label, eyes_pos):
        '''
        Detects the pupil or iris in the given frame using thresholding and contour analysis.

        Arguments:
        - frame: The input image frame in which the pupil/iris is to be detected.

        Returns:
        - A numpy array containing the coordinates of the center of the detected pupil/iris, 
          or (None, None) if no contours are found.
        '''

        # Select iris and pupil regions
        pupil_pixels = mask == label["pupil"]
        iris_pixels = mask == label["iris"]

        # Merge the masks
        merged_mask = (pupil_pixels | iris_pixels).astype(np.uint8)

        #cv2.imshow(f'Image {eyes_pos}', merged_mask*255)

        # Find the contours of the pupil/iris mask
        contours, _ = cv2.findContours(merged_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Display the original image with contours
        frame_with_contours = frame.copy()
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)
        #cv2.imshow('Contours', frame_with_contours)

        # Control the number of contours found
        if len(contours) == 0:
            return (None, None)
        
        # Sort the contours in descending order of their area
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # Sort in descending order of the list of contours by their area
        cv2.drawContours(frame, [contours[0]], -1, (0, 255, 0), 2)
        #cv2.imshow('Largest Contour', frame)

        # Select the contour with the biggest area
        largest_contour = contours[0]

        # Check that the contour has enough points to fit an ellipse (at least 5)
        if len(largest_contour) >= 5:
            # Fit an ellipse to the largest contour
            ellipse = cv2.fitEllipse(largest_contour)
            
            # Validate ellipse dimensions
            (center, axes, angle) = ellipse
            (major_axis, minor_axis) = axes
            
            if major_axis > 0 and minor_axis > 0:
                # Draw the ellipse on the original frame
                cv2.ellipse(frame_with_contours, ellipse, (255, 0, 0), 2)
                               
                
                # The center of the ellipse is the center of the pupil/iris
                center = np.array([int(center[0]), int(center[1])], dtype=np.int32)
            else:
                center = (None, None)  # Invalid ellipse dimensions

        else:
            center = (None, None) # Invalid ellipse dimensions

        cv2.waitKey(1)

        return center  # Return the center of the pupil/iris                   
       
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
            return None,None
        # Convert relative position to absolute position based on the ROI
        return int(roi[0] + relative_position[0]), int(roi[1] + relative_position[1])