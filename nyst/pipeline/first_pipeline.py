import cv2
import numpy as np

from nyst.roi import FirstRegionSelector, FirstEyeRoiDetector, FirstEyeRoiSegmenter
from nyst.utils import FirstLatch
from nyst.pupil import ThresholdingPupilDetector
from nyst.analysis import FirstSpeedExtractor
from nyst.visualization import FirstFrameAnnotator

class FirstPipeline:
    def __init__(self):
        self.region_selector = FirstRegionSelector()
        self.eye_roi_detector = FirstEyeRoiDetector("yolov8")
        self.eye_roi_segmenter = FirstEyeRoiSegmenter('nyst/seg_eyes/model.h5')
        self.left_eye_roi_latch = FirstLatch()
        self.right_eye_roi_latch = FirstLatch()
        self.pupil_detector = ThresholdingPupilDetector(threshold=50)
        self.frame_annotator = FirstFrameAnnotator()
        self.speed_extractor = FirstSpeedExtractor()
        
    def apply(self, frame, count_from_lastRoiupd, update_roi=True):
        # Uptdating eyes ROI
        if update_roi and count_from_lastRoiupd < 3000:
            try:
                # Calculate the ROI of left and right eyes
                rois = self.eye_roi_detector.apply(frame)

                # Unpack the ROIs and assign them individually to two separate variables
                left_eye_roi = rois.get_left_eye_roi()
                right_eye_roi = rois.get_right_eye_roi()

                # Save the ROIs to the latch variables in order to have two distinct pipeline blocks
                self.left_eye_roi_latch.set(left_eye_roi)
                self.right_eye_roi_latch.set(right_eye_roi)
                count_from_lastRoiupd = 0
                print("normal", count_from_lastRoiupd, end="\t\t")
            except Exception as e:
                count_from_lastRoiupd+=1
                print("exception", count_from_lastRoiupd, end="\t\t")
                print(e)
            
        else:
            raise RuntimeError('Unable to find a face in the last 30fps')
            
        # Get distinct ROIs value and save them to two specific variables
        left_eye_roi = self.left_eye_roi_latch.get()
        right_eye_roi = self.right_eye_roi_latch.get()

        if left_eye_roi is None and right_eye_roi is None:
            return (None, None), (None, None), count_from_lastRoiupd
        
        # Apply ROI to the selected frame and assign the result to specific variables
        left_eye_frame = self.region_selector.apply(frame, left_eye_roi)
        right_eye_frame = self.region_selector.apply(frame, right_eye_roi)

        if left_eye_frame.shape[0] == 0 or left_eye_frame.shape[1] == 0 or right_eye_frame.shape[0] == 0 or right_eye_frame.shape[1] == 0:
            return (None, None), (None, None), count_from_lastRoiupd
        # cv2.imshow('Left eye box',left_eye_frame)
        # cv2.imshow('Right eye box',right_eye_frame)

        # Apply segmented ROI to the selected frame and assign the result to specific variables
        left_eye_frame = self.eye_roi_segmenter.apply(left_eye_frame)
        right_eye_frame = self.eye_roi_segmenter.apply(right_eye_frame)
        # cv2.imshow('Left eye segmented',left_eye_frame)
        # cv2.imshow('Right eye segmented',right_eye_frame)

        # 
        left_pupil_relative_position = self.pupil_detector.apply(left_eye_frame, "left_treshold")
        right_pupil_relative_position = self.pupil_detector.apply(right_eye_frame, "left_treshold")

        left_pupil_absolute_position = self.region_selector.relative_to_absolute(left_pupil_relative_position, left_eye_roi)
        right_pupil_absolute_position = self.region_selector.relative_to_absolute(right_pupil_relative_position, right_eye_roi)

        return left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd

    
    def run(self, video_path):
        # Initialize lists to store absolute positions of left and right eye pupils
        left_eye_absolute_positions = []
        right_eye_absolute_positions = []

        # Open the video file   
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (FPS) and resolution of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        '''
        if self.video_preprocessing.invert_resolution:
            resolution = resolution[::-1]
        '''


        # Create a video writer object to save the annotated video
        annotated_video_writer = cv2.VideoWriter("annotated_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
        count_from_lastRoiupd = 0

        # Read the first frame of the video
        ret, frame = cap.read()
        if ret is False:
            # Raise an error if the frame could not be read
            raise RuntimeError("Error reading video")

        # Apply the processing method for absolute position pupil estimation to the frame
        left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame,count_from_lastRoiupd,update_roi=True)

        # Append the positions to the respective lists
        left_eye_absolute_positions.append(left_pupil_absolute_position)
        right_eye_absolute_positions.append(right_pupil_absolute_position)
        
        # Annotate the frame with the pupil positions
        annotated_frame = self.frame_annotator.apply(frame, left_pupil_absolute_position, right_pupil_absolute_position)

        # Display the annotated frame
        # cv2.imshow("frame", annotated_frame)
        cv2.waitKey(1)

        # Write the annotated frame to the video writer
        annotated_video_writer.write(annotated_frame)


        # Loop to process each frame of the video
        while True:
        
            # Read the next frame of the video
            ret, frame = cap.read()
            # Break the loop if no frame is read (end of video)
            if ret is False:
                break

            # Apply the processing method to the frame
            left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame,count_from_lastRoiupd)

            # Append the positions to the respective lists
            left_eye_absolute_positions.append(left_pupil_absolute_position)
            right_eye_absolute_positions.append(right_pupil_absolute_position)
            print(left_pupil_absolute_position, right_pupil_absolute_position) #Print the positions

            # Annotate the frame with the pupil positions
            annotated_frame = self.frame_annotator.apply(frame, left_pupil_absolute_position, right_pupil_absolute_position)

            # Display the annotated frame
            # cv2.imshow("frame", annotated_frame)

            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Write the annotated frame to the video writer
            annotated_video_writer.write(annotated_frame)

        # Clean up and release resources
        cv2.destroyAllWindows()
        cap.release()
        annotated_video_writer.release()

        # Convert the positions lists to numpy arrays
        left_eye_absolute_positions = np.array(left_eye_absolute_positions, dtype=np.float32)
        right_eye_absolute_positions = np.array(right_eye_absolute_positions, dtype=np.float32)

        # Extract speed information for the left and right eyes
        left_eye_speed_dict = self.speed_extractor.apply(left_eye_absolute_positions, fps)
        right_eye_speed_dict = self.speed_extractor.apply(right_eye_absolute_positions, fps)

        # Combine the speed information into a dictionary
        speed_dict = {
            resolution: {"left": left_eye_speed_dict[resolution], "right": right_eye_speed_dict[resolution]}
            for resolution in left_eye_speed_dict
            }
        
        # Create the output dictionary with positions and speed information
        output_dict = {
            "position": {
                "left": left_eye_absolute_positions,
                "right": right_eye_absolute_positions
            },
            "speed": speed_dict
        }

        return output_dict
    