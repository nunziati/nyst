# Class that provides the functions to execute the calculation of the eye ROI
class FirstRoi:
    """
    Class that encapsulates the Regions of Interest (ROI) for the left and right eyes.

    Attributes:
    - left_eye_roi: The ROI box for the left eye.
    - right_eye_roi: The ROI box for the right eye.
    """
    def __init__(self, left_eye_roi, right_eye_roi):
        self.left_eye_roi = left_eye_roi
        self.right_eye_roi = right_eye_roi
    
    def get_left_eye_roi(self):
        '''
        Retrieves the ROI coordinates for the left eye.

        Returns:
        - The ROI coordinates for the left eye.
        '''
        return self.left_eye_roi
    
    def get_right_eye_roi(self): 
        '''
        Retrieves the ROI coordinates for the right eye.

        Returns:
        - The ROI coordinates for the right eye.
        '''
        return self.right_eye_roi