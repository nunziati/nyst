import cv2

class FirstFrameAnnotator:
    """
    Class responsible for annotating frames by drawing crosshairs on the detected pupil positions.
    """
    def __init__(self):
        pass

    def apply(self, frame, left_pupil_absolute_position, right_pupil_absolute_position, length:int=50):
        """
        Annotates the given frame with crosshairs at the positions of the left and right pupils.

        Arguments:
        - frame: The image frame  to be annotated.
        - left_pupil_absolute_position: A tuple (x, y) representing the absolute position of the left pupil in the frame.
        - right_pupil_absolute_position: A tuple (x, y) representing the absolute position of the right pupil in the frame.
        - length: The length of the crosshairs to be drawn. Default is 50 pixels.

        Returns:
        - The annotated frame with crosshairs drawn at the positions of the left and right pupils.
        """
        # Get the dimensions of the frame
        rows, cols, _ = frame.shape

        # Extract and convert the x and y coordinates of the left and right pupils to integers
        left_x, left_y = map(int, left_pupil_absolute_position)
        right_x, right_y = map(int, right_pupil_absolute_position)
        
        # Draw crosshairs on the left pupil if coordinates are available
        if left_x is not None and left_y is not None:
            cv2.line(frame, (max(left_x - length // 2, 0), left_y), (min(left_x + length // 2, cols), left_y), (0, 255, 0), 2)
            cv2.line(frame, (left_x, max(left_y - length // 2, 0)), (left_x, min(left_y + length // 2, rows)), (0, 255, 0), 2)

        # Draw crosshairs on the right pupil if coordinates are available
        if right_x is not None and right_y is not None:
            cv2.line(frame, (max(right_x - length // 2, 0), right_y), (min(right_x + length // 2, cols), right_y), (0, 255, 0), 2)
            cv2.line(frame, (right_x, max(right_y - length // 2, 0)), (right_x, min(right_y + length // 2, rows)), (0, 255, 0), 2)

        return frame # Return the annotated frame
