# Class that provides the functions to execute the calculation of the eye ROI

class FirstRoi:

    def __init__(self, left_eye_roi, right_eye_roi):
        self.left_eye_roi = left_eye_roi
        self.right_eye_roi = right_eye_roi
    
    def get_left_eye_roi(self): # Function to apply the ROI calculation of the left eye
        return self.left_eye_roi
    
    def get_right_eye_roi(self): # Function to apply the ROI calculation of the right eye
        return self.right_eye_roi