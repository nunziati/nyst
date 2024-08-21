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
        #cv2.imshow('Image', frame)

        # Transform the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('Image gray', gray_frame)
        
        # Take a copy of the frame
        gray_frame_copy = gray_frame.copy()

        # Increase frame contrast
        alpha = 2.2  # Contrast factor
        beta = 0.     # Lightness factor
        high_contrast_gray = cv2.convertScaleAbs(gray_frame_copy, alpha=alpha, beta=beta)
        #cv2.imshow('High Contrast Gray', high_contrast_gray)

        '''# Aumentare il contrasto
        #equalized_gray_frame = cv2.equalizeHist(gray_frame_copy)

        # Calcolare la media dei pixel diversi dal bianco
        non_white_pixels = high_contrast_gray[high_contrast_gray < 255]
        #mean_value = np.mean(non_white_pixels) if len(non_white_pixels) > 0 else 50'''
        
        # Calculate the median between distinct gray levels excluding white to set a correct threshold
        non_white_pixels = high_contrast_gray[high_contrast_gray < 255] # All pixel with gray levels different from white
        unique_gray_levels = np.unique(non_white_pixels) # Gray levels excluding white
        median_value = np.median(unique_gray_levels) if len(unique_gray_levels) > 0 else 50 # Median

        # Median percentage variation
        perc_var = 0.45

        # Add a small value to the median value to correctly estimate a threshold
        threshold_value = median_value - (perc_var*median_value)

        # Apply a Gaussian blur to the grayscale frame
        blurred_roi = cv2.GaussianBlur(high_contrast_gray, (7, 7), 0) # 7x7: dimension of the kernel, 0: "0" indicates that the standard deviation will be automatically calculated based on the size of the kernel

        # Return the pupil/iris mask
        _, threshold = cv2.threshold(blurred_roi, threshold_value, 255, cv2.THRESH_BINARY_INV)
        #cv2.imshow('Threshold Image', threshold)

        # Find the contours of the pupil/iris mask
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        # Find the contour with the biggest area
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
    
    
    def apply_2(self, frame):
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
        cv2.imshow('Threshold Image', threshold)

        # Find the contours of the pupil/iris mask
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Visualizzare l'immagine originale con i contorni
        frame_with_contours = frame.copy()
        cv2.drawContours(frame_with_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Contours', frame_with_contours)


        # Control the number of contours found
        if len(contours) == 0:
            return (None, None)
        
        # Sort the contours in descending order of their area
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True) # Sort in descending order of the list of contours by their area
        
        # Find the box that contain the contours with the largest area
        (x_min, y_min, w, h) = cv2.boundingRect(contours[0])

        # Visualizzare il rettangolo di delimitazione
        cv2.rectangle(frame_with_contours, (x_min, y_min), (x_min + w, y_min + h), (255, 0, 0), 2)
        cv2.imshow('Bounding Box', frame_with_contours)

        # Mostrare tutte le finestre fino a quando non viene premuto un tasto
        cv2.waitKey(1)

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
            return None,None
        # Convert relative position to absolute position based on the ROI
        return roi[0] + relative_position[0], roi[1] + relative_position[1]