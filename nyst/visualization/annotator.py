import cv2

class FirstFrameAnnotator:
    def __init__(self):
        pass

    def apply(self, frame, left_pupil_absolute_position, right_pupil_absolute_position, length=50):
        rows, cols, _ = frame.shape

        left_x, left_y = left_pupil_absolute_position
        right_x, right_y = right_pupil_absolute_position
        
        if left_x is not None and left_y is not None:
            cv2.line(frame, (max(left_x - length // 2, 0), left_y), (min(left_x + length // 2, cols), left_y), (0, 255, 0), 2)
            cv2.line(frame, (left_x, max(left_y - length // 2, 0)), (left_x, min(left_y + length // 2, rows)), (0, 255, 0), 2)

        if right_x is not None and right_y is not None:
            cv2.line(frame, (max(right_x - length // 2, 0), right_y), (min(right_x + length // 2, cols), right_y), (0, 255, 0), 2)
            cv2.line(frame, (right_x, max(right_y - length // 2, 0)), (right_x, min(right_y + length // 2, rows)), (0, 255, 0), 2)

        return frame
