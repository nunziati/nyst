import cv2
import numpy as np
import os
import csv
from nyst.roi import FirstRegionSelector, FirstEyeRoiDetector, FirstEyeRoiSegmenter
from nyst.utils import FirstLatch
from nyst.pupil import ThresholdingPupilDetector
from nyst.analysis import FirstSpeedExtractor
from nyst.visualization import FirstFrameAnnotator

class FirstPipeline:
    def __init__(self):
        self.region_selector = FirstRegionSelector()
        self.eye_roi_detector = FirstEyeRoiDetector("yolov8")
        self.eye_roi_segmenter = FirstEyeRoiSegmenter('D:/model.h5')
        self.left_eye_roi_latch = FirstLatch()
        self.right_eye_roi_latch = FirstLatch()
        self.pupil_detector = ThresholdingPupilDetector(threshold=50)
        self.frame_annotator = FirstFrameAnnotator()
        self.speed_extractor = FirstSpeedExtractor()
        
    def apply(self, frame, count_from_lastRoiupd, idx_frame, threshold=100, update_roi=True):
        # Update the eye ROI if specified and if the count is less than a threshold
        if update_roi and count_from_lastRoiupd < threshold:
            try:
                # Compute the ROI for the left and right eyes
                rois = self.eye_roi_detector.apply(frame, idx_frame)
    
                # Unpack the ROIs for the left and right eyes
                left_eye_roi = rois.get_left_eye_roi()
                right_eye_roi = rois.get_right_eye_roi()
               
                # Save the ROIs to latch variables to have two distinct pipeline blocks
                self.left_eye_roi_latch.set(left_eye_roi)
                self.right_eye_roi_latch.set(right_eye_roi)
                
                count_from_lastRoiupd = 0 # Counter last latch update
                # print("Normal: ", count_from_lastRoiupd, end="\t\t")
            except Exception as e:
                # Increment count and print exception details if an error occurs
                count_from_lastRoiupd+=1
                print("Exception: ", count_from_lastRoiupd, end="\t\t")
                print(e)
            
        else:
            raise RuntimeError('Unable to find a face in the last 30fps')
        
        # Retrieve the ROI values from the latch variables
        left_eye_roi = self.left_eye_roi_latch.get()
        right_eye_roi = self.right_eye_roi_latch.get()
        
        # Return if both ROIs are None
        if left_eye_roi is None and right_eye_roi is None:
            return (None, None), (None, None), count_from_lastRoiupd
        
        # Apply ROI to the selected frame and store the results
        left_eye_frame = self.region_selector.apply(frame, left_eye_roi)
        right_eye_frame = self.region_selector.apply(frame, right_eye_roi)
        
        # Check if the frames are empty and return None
        if left_eye_frame.shape[0] == 0 or left_eye_frame.shape[1] == 0 or right_eye_frame.shape[0] == 0 or right_eye_frame.shape[1] == 0:
            return (None, None), (None, None), count_from_lastRoiupd
        # Show the frames with the detected eye ROIs
        # cv2.imshow('Left eye box',left_eye_frame)
        # cv2.imshow('Right eye box',right_eye_frame)
       
        # Apply segmentation to the eye frames ROI
        left_eye_frame = self.eye_roi_segmenter.apply(left_eye_frame)
        right_eye_frame = self.eye_roi_segmenter.apply(right_eye_frame)
        # Show the segmented eye of the frames
        # cv2.imshow('Left eye segmented',left_eye_frame)
        # cv2.imshow('Right eye segmented',right_eye_frame)

        # Detect the relative position of the pupil in each eye frame
        left_pupil_relative_position = self.pupil_detector.apply(left_eye_frame, "left_treshold")
        right_pupil_relative_position = self.pupil_detector.apply(right_eye_frame, "left_treshold")
        
        # Convert the relative pupil positions to absolute positions based on the ROI 
        left_pupil_absolute_position = self.pupil_detector.relative_to_absolute(left_pupil_relative_position, left_eye_roi) # (X,Y) Absolute position of the left eye
        right_pupil_absolute_position = self.pupil_detector.relative_to_absolute(right_pupil_relative_position, right_eye_roi) # (X,Y) Absolute position of the right eye
        
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
        
        # Frame counter
        count = 0

        if ret is False:
            # Raise an error if the frame could not be read
            raise RuntimeError("Error reading video")
        
        # Apply the processing method for absolute position pupil estimation to the frame
        left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame,count_from_lastRoiupd,count)   # PROBLEMA PARTE DA QUI

        # Append the positions to the respective lists (list of tuple of absolute x,y coordinates)
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
            # Increment the frame counter
            count += 1
            
            # Print the frame counter
            print("Frame:", count)
            # Apply the processing method to the frame
            left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame,count_from_lastRoiupd,count)
           
            # Append the positions to the respective lists
            left_eye_absolute_positions.append(left_pupil_absolute_position)
            right_eye_absolute_positions.append(right_pupil_absolute_position)
            #print(left_pupil_absolute_position, right_pupil_absolute_position) #Print the positions

            # Annotate the frame with the pupil positions
            annotated_frame = self.frame_annotator.apply(frame, left_pupil_absolute_position, right_pupil_absolute_position)

            # Display the annotated frame
            #cv2.imshow("frame", annotated_frame)

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
        
        print('\n\----->   Video Features Extracted')
        
        return output_dict
    
    def videos_feature_extractor(self, input_folder, output_path):
        # Path for the video_features CSV file
        output_features_path = os.path.join(output_path, 'video_features.csv')
        # Verify if the file already exists
        file_exists = os.path.isfile(output_features_path)
        
        # Prepare the CSV writer for the labels file
        with open(output_features_path, 'a', newline='') as csvfile:
            # Write the video_features CSV file
            writer = csv.writer(csvfile)
            # Write header only if file does not exist
            if not file_exists:
                writer.writerow([
                    'video', 'resolution', 'left_position X', 'left_position Y', 
                    'right_position X', 'right_position Y', 'left_speed X', 'left_speed Y', 
                    'right_speed X', 'right_speed Y'
                ])  

            # Iterate through all video files in the input folder
            for video in os.listdir(input_folder):
                if video.endswith('.mp4'):  # Add other video formats if needed
                    video_path = os.path.join(input_folder, video)
                    try:
                        # Run the processing on the video
                        output_dict = self.run(video_path)

                        # Create a unique path for the output video name
                        output_video_relative_path = os.path.normpath(os.path.join("videos", video)) # Create the relative path for the output video

                        # Extract positions and speed from the output_dict
                        left_positions = output_dict['position']['left']
                        right_positions = output_dict['position']['right']
                        speed_dict = output_dict['speed']

                        # Write the result to the labels CSV file, including resolution
                        for resolution in speed_dict:
                            # Initialisation lists
                            left_speed_x = []
                            left_speed_y = []
                            right_speed_x = []
                            right_speed_y = []

                            # Store the speeds
                            for i in range(len(left_positions)):
                                left_speed_x.append(speed_dict[resolution]["left"][i][0])
                                left_speed_y.append(speed_dict[resolution]["left"][i][1])
                                right_speed_x.append(speed_dict[resolution]["right"][i][0])
                                right_speed_y.append(speed_dict[resolution]["right"][i][1])

                            # Store lists as strings in CSV
                            writer.writerow([
                                output_video_relative_path, resolution,
                                [pos[0] for pos in left_positions],
                                [pos[1] for pos in left_positions],
                                [pos[0] for pos in right_positions],
                                [pos[1] for pos in right_positions],
                                left_speed_x,
                                left_speed_y,
                                right_speed_x,
                                right_speed_y
                            ])
                    except Exception as e:
                        print(f"Failed to process {video}: {e}")

            # Completion message
            print("Video processing completed successfully.")

