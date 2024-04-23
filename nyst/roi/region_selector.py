class FirstRegionSelector:
    def __init__(self):
        pass

    def apply(self, frame, roi): # Crop the frame with the specified ROI
        return frame[roi[1]:roi[3], roi[0]:roi[2]]
    
    def relative_to_absolute(self, relative_position, roi):
        if relative_position[0] is None or relative_position[1] is None:
            return None, None
        
        return roi[0] + relative_position[0], roi[1] + relative_position[1]