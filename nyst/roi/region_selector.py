class FirstRegionSelector:
    def __init__(self):
        pass # Initialize the class without any specific attributes

    # Crop the frame with the specified Region Of Interest (ROI)
    def apply(self, frame, roi):
        return frame[roi[1]:roi[3], roi[0]:roi[2]]