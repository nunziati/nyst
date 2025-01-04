class FirstRegionSelector:
    def __init__(self):
        pass # Initialize the class without any specific attributes

    # Crop the frame with the specified Region Of Interest (ROI)
    def apply(self, frame, roi):
        return frame[int(roi[1]):int(roi[3]), int(roi[0]):int(roi[2])]