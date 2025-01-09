import cv2
import numpy as np
import os
import csv
import sys
import traceback

# Add the 'code' directory to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.roi import FirstRegionSelector, FirstEyeRoiDetector, FirstEyeRoiSegmenter
from nyst.utils import FirstLatch
from nyst.pupil import ThresholdingPupilDetector
from nyst.analysis import FirstSpeedExtractor
from nyst.visualization import FirstFrameAnnotator
from nyst.preprocessing import PreprocessingSignalsVideos

class FirstPipeline:
    def __init__(self):
        self.region_selector = FirstRegionSelector()
        self.eye_roi_detector = FirstEyeRoiDetector("/repo/porri/nyst/yolo_models/best_yolo11m.pt")
        self.left_eye_roi_latch = FirstLatch()
        self.right_eye_roi_latch = FirstLatch()
        self.left_eye_center_latch = FirstLatch()
        self.right_eye_center_latch = FirstLatch()
        self.eye_roi_segmenter = FirstEyeRoiSegmenter('/repo/porri/model.h5')
        self.pupil_detector = ThresholdingPupilDetector(threshold=50)
        self.preprocess = PreprocessingSignalsVideos()
        self.frame_annotator = FirstFrameAnnotator()
        self.speed_extractor = FirstSpeedExtractor()
        
    def apply(self, frame, count_from_lastRoiupd:int, count:int, threshold:int=30, update_roi:bool=True) -> tuple:
        '''
        Applies the eye detection and pupil position extraction on a given video frame.

        Arguments:
        - frame: The current video frame to process.
        - count_from_lastRoiupd (int): The counter indicating the number of frames since the last ROI update.
        - idx_frame (int): The index of the current frame in the video sequence.
        - threshold (int): The maximum number of frames to wait before forcing an ROI update (default is 100).
        - update_roi (bool): A boolean flag indicating whether to update the eye ROI (default is True).

        Returns:
        - left_pupil_absolute_position: The absolute (x, y) position of the left pupil in the frame.
        - right_pupil_absolute_position: The absolute (x, y) position of the right pupil in the frame.
        - count_from_lastRoiupd: The updated counter indicating the number of frames since the last ROI update.
        '''

        '''# CONTROL 1 #
        print('================================ CONTROL STEP 1 =================================')
        print(f"Frame type: {type(frame)}, Frame shape: {getattr(frame, 'shape', 'N/A')}")
        print(f"count_from_lastRoiupd: {count_from_lastRoiupd}, threshold: {threshold}, update_roi: {update_roi}")'''
       
        # Update the eye ROI if specified and if the count is less than a threshold
        if update_roi and count_from_lastRoiupd < threshold:
            try:
                # Retrieve last value
                if self.left_eye_roi_latch.get() is not None:
                    old_left_eye_roi = self.left_eye_roi_latch.get()
                    old_right_eye_roi = self.right_eye_roi_latch.get()
                else:
                    self.left_eye_roi_latch.set(np.zeros(4))
                    self.right_eye_roi_latch.set(np.zeros(4))
                    old_left_eye_roi = self.left_eye_roi_latch.get()
                    old_right_eye_roi = self.right_eye_roi_latch.get()

                # Compute the ROI for the left and right eyes
                left_eye_roi, right_eye_roi, old_left_eye_roi, old_right_eye_roi, count_from_lastRoiupd = self.eye_roi_detector.apply(frame, count_from_lastRoiupd, old_left_eye_roi, old_right_eye_roi)
    
               
                # Save the ROIs to latch variables to have two distinct pipeline blocks
                self.left_eye_roi_latch.set(old_left_eye_roi)
                self.right_eye_roi_latch.set(old_right_eye_roi)

            except Exception as e:
                # Increment count and print exception details if an error occurs
                count_from_lastRoiupd+=1
                print("Exception: ", count_from_lastRoiupd, end="\t\t")
                print(e)
            
        else:
            raise RuntimeError(f'Unable to find a face in the last {threshold} frames')
        
        # Apply ROI to the selected frame and store the results
        left_eye_frame = self.region_selector.apply(frame, left_eye_roi)
        right_eye_frame = self.region_selector.apply(frame, right_eye_roi)

        '''# CONTROL #
        print('================================ CONTROL STEP 2 =================================')
        # Frame size control
        if frame is not None and len(frame.shape) == 3:
            rows, cols, _ = frame.shape
            print(f"Frame dimensions: {rows}x{cols}")
        else:
            raise ValueError("Frame non valido o privo di dimensioni.")
        
        # Verification of region selection result
        if left_eye_frame is None or right_eye_frame is None:
            raise ValueError("La selezione della regione ha restituito un frame nullo.")

        print(f"Left eye frame shape: {left_eye_frame.shape}")
        print(f"Right eye frame shape: {right_eye_frame.shape}")'''
        
        # Checking validity of ROIs
        if not all(isinstance(roi, np.ndarray) for roi in [left_eye_roi, right_eye_roi]):
            raise ValueError("Le ROI devono essere array NumPy.")

        if not all(np.issubdtype(roi.dtype, np.number) for roi in [left_eye_roi, right_eye_roi]):
            print(f"ROI values - Left: {left_eye_roi}, Right: {right_eye_roi}")
            raise ValueError("I valori delle ROI devono essere numerici.")

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
        left_pupil_relative_position = self.pupil_detector.apply(left_eye_frame,count)
        right_pupil_relative_position = self.pupil_detector.apply(right_eye_frame,count)
        
        # Convert the relative pupil positions to absolute positions based on the ROI 
        if left_pupil_relative_position[0] is not None and left_pupil_relative_position[1] is not None:
            left_pupil_absolute_position = self.pupil_detector.relative_to_absolute(left_pupil_relative_position, left_eye_roi) # (X,Y) Absolute position of the left eye
            # Save the absolute centers to the latch variables
            self.left_eye_center_latch.set(left_pupil_absolute_position)
        else:
            left_pupil_absolute_position =  self.left_eye_center_latch.get()
            print("Get the left center correct position.")
        
        if right_pupil_relative_position[0] is not None and right_pupil_relative_position[1] is not None:
            right_pupil_absolute_position = self.pupil_detector.relative_to_absolute(right_pupil_relative_position, right_eye_roi) # (X,Y) Absolute position of the right eye
            # Save the absolute centers to the latch variables
            self.right_eye_center_latch.set(right_pupil_absolute_position)
        else:
            right_pupil_absolute_position =  self.right_eye_center_latch.get()
            print("Get the right center correct position.")

        '''# CONTROL #
        print('================================ CONTROL STEP 3 =================================')

        print(f"Left pupil position: {left_pupil_absolute_position}")
        print(f"Right pupil position: {right_pupil_absolute_position}")'''

        
        return left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd

    def run(self, video_path:str, output_path:str, idx:int) -> dict:
        '''
        Processes a video to extract the absolute positions of the left and right eye pupils,
        annotates each frame, and calculates the speed of pupil movements.

        Arguments:
        - video_path (str): The path to the video file to be processed.

        Returns:
        - output_dict (dict): A dictionary containing the extracted positions and speed information for the left and right eye pupils.
        '''
        # Initialize lists to store absolute positions of left and right eye pupils
        left_eye_absolute_positions = []
        right_eye_absolute_positions = []

        # Creare la cartella solo se non esiste già
        os.makedirs(f"{output_path}/Annotated_videos", exist_ok=True)

        # Open the video file   
        cap = cv2.VideoCapture(video_path)

        # Get the frames per second (FPS) and resolution of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        resolution = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
   
        # Create a video writer object to save the annotated video
        annotated_video_writer = cv2.VideoWriter(f"{output_path}/Annotated_videos/annotated_video_{idx}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, resolution)
        count_from_lastRoiupd = 0

        # Read the first frame of the video
        ret, frame = cap.read()
        
        # Frame counter
        count = 0

        # Print the frame counter
        print("\n\nFrame: 0  ---------------------------------------------------------------")

        if ret is False:
            # Raise an error if the frame could not be read
            raise RuntimeError("Error reading video")
        
        # Apply the processing method for absolute position pupil estimation to the frame
        left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame,count_from_lastRoiupd,count)

        # Append the positions to the respective lists (list of tuple of absolute x,y coordinates)
        left_eye_absolute_positions.append(left_pupil_absolute_position)
        right_eye_absolute_positions.append(right_pupil_absolute_position)
        
        # Annotate the frame with the pupil positions
        annotated_frame = self.frame_annotator.apply(frame, left_pupil_absolute_position, right_pupil_absolute_position)
        
        # Display the annotated frame
        #cv2.imshow("frame", annotated_frame)
        #cv2.waitKey(1)

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
            print(f"\n\nFrame: {count} ---------------------------------------------------------------")
           
            # Apply the processing method to the frame
            try:
                left_pupil_absolute_position, right_pupil_absolute_position, count_from_lastRoiupd = self.apply(frame, count_from_lastRoiupd, count)
            except Exception as e:
                print(f"Errore durante l'elaborazione del frame {count}: {e}")
                traceback.print_exc()
                continue
        
            # Append the positions to the respective lists
            left_eye_absolute_positions.append(left_pupil_absolute_position)
            right_eye_absolute_positions.append(right_pupil_absolute_position)
            print(left_pupil_absolute_position, right_pupil_absolute_position) #Print the positions

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
        left_eye_absolute_positions_dirty = np.array(left_eye_absolute_positions)
        right_eye_absolute_positions_dirty = np.array(right_eye_absolute_positions)

        # Ensures nan values are properly handled
        left_eye_absolute_positions = self.preprocess.interpolate_nans(left_eye_absolute_positions_dirty)
        right_eye_absolute_positions = self.preprocess.interpolate_nans(right_eye_absolute_positions_dirty)

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
    
    # xtracts features from all videos
    def videos_feature_extractor(self, input_folder:str, output_path:str) -> None:
        '''
        Extracts features from videos in a folder and saves them to a CSV file.

        Arguments:
        - input_folder (str): The path to the folder containing the videos to be processed.
        - output_path (str): The path to the folder where to save the CSV file with the extracted features.

        Returns:
        - Saves the features in a CSV file.
        '''
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
            for idx, video in enumerate(os.listdir(input_folder)):
                if video.endswith('.mp4'):  # Add other video formats if needed
                    video_path = os.path.join(input_folder, video)
                    try:
                        print(f'\n\nFeature extraction of the video: {video} ----- Video: {idx}')
                        # Run the processing on the video
                        output_dict = self.run(video_path, output_path, idx)

                        # Salva i risultati del PupilDetector in un file
                        self.pupil_detector.save_threshold_counts("/repo/porri/nyst_labelled_videos/threshold_counts.txt", video)
                        print("Saved threshold counts to threshold_counts.txt.")

                        # Create a unique path for the output video name
                        output_video_relative_path = os.path.normpath(os.path.join("videos", video)) # Create the relative path for the output video

                        # Extract positions and speed from the output_dict
                        left_positions = output_dict['position']['left']
                        right_positions = output_dict['position']['right']
                        speed_dict = output_dict['speed']

                        # Write the result to the labels CSV file, including resolution
                        for resolution in speed_dict:
                            # Convert numpy arrays to lists (if needed) and then to strings
                            left_positions_x = str([pos[0] for pos in left_positions])
                            left_positions_y = str([pos[1] for pos in left_positions])
                            right_positions_x = str([pos[0] for pos in right_positions])
                            right_positions_y = str([pos[1] for pos in right_positions])

                            left_speed_x = str([speed_dict[resolution]["left"][i][0] for i in range(len(left_positions))])
                            left_speed_y = str([speed_dict[resolution]["left"][i][1] for i in range(len(left_positions))])
                            right_speed_x = str([speed_dict[resolution]["right"][i][0] for i in range(len(right_positions))])
                            right_speed_y = str([speed_dict[resolution]["right"][i][1] for i in range(len(right_positions))])

                            # Store lists as strings in CSV
                            writer.writerow([
                                output_video_relative_path, resolution,
                                left_positions_x,
                                left_positions_y,
                                right_positions_x,
                                right_positions_y,
                                left_speed_x,
                                left_speed_y,
                                right_speed_x,
                                right_speed_y
                            ])
                    except Exception as e:
                        print(f"Failed to process {video}: {e}")
                        # Print the detailed traceback
                        traceback.print_exc()

            # Completion message
            print("\nVideo processing completed successfully.")

