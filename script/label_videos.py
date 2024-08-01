import os
import pickle
from collections import deque
import csv
import argparse

import cv2

# Define supported video file extensions
video_extensions = ['.mp4', '.mkv']

# Function to parse command-line arguments with default values
def parse_args(default_args):
    root = default_args['input_dir']
    output_path = default_args['output_dir']
    clip_duration = default_args['clip_duration']
    overlapping = default_args['overlapping']

    parser = argparse.ArgumentParser(description='Label videos')
    parser.add_argument('--input_dir', type=str, default=root, help='Input directory')
    parser.add_argument('--output_dir', type=str, default=output_path, help='Output directory')
    parser.add_argument('--clip_duration', type=int, default=clip_duration, help='Clip duration in seconds')
    parser.add_argument('--overlapping', type=int, default=overlapping, help='Overlapping in seconds')

    return parser.parse_args()

# Main function to handle videos labelling process
def main():

    ### VIDEO SETTINGS ###

    # Default settings
    root = '/repo/porri/dataset_video'
    output_path = '/repo/porri/output_videos'
    clip_duration = 10 # in seconds
    overlapping = 8 # in seconds

    # Default arguments dictionary
    default_args = { 'input_dir': root, 'output_dir': output_path, 'clip_duration': clip_duration, 'overlapping': overlapping }

    # Parse command-line arguments
    args = parse_args(default_args)

    root = args.input_dir
    output_path = args.output_dir
    clip_duration = args.clip_duration
    overlapping = args.overlapping

    
    ### LABELLING ###
    
    # List to store video labels
    labels = [] 

    # Create output directory for videos if it doesn't exist
    os.makedirs(os.path.join(output_path, "videos"), exist_ok=True)

    # Get the list of ordered video names from the input directory
    videos = os.listdir(root)
    videos = sorted([video for video in videos if any(video.endswith(ext) for ext in video_extensions)])
    
    # Loop through the list of videos
    for _, video in enumerate(videos):
        
        # Window to visualize the video 
        cv2.namedWindow('Frame')
        cv2.resizeWindow('Frame', 800, 600)
        
        # Create the video path
        video_path = os.path.join(root, video)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise("Error opening video")
        
        fps = cap.get(cv2.CAP_PROP_FPS) # Get number of fps
        clip_frames = round(clip_duration * fps) # Number of frames per clip

        # Queue to store overlapping frames
        overlapping_queue = deque()
        clip_id = 1

        while True:
            output_video_name = video.split('.')[0] + f'_{str(clip_id).rjust(3, "0")}.mp4' # Generate the output video file name
            output_video_relative_path = os.path.join("videos", output_video_name) # Create the relative path for the output video
            output_video_path = os.path.join(output_video_relative_path, output_video_name) # Create the full path for the output video
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4)))) # Initialize the video writer object
            
            # Flags
            end_of_video = False # Flag to indicate if the end of the video has been reached
            video_ended = True # Flag to indicate if the video ended properly

            # Loop over the frames in a clip
            for i in range(clip_frames):
                if clip_id > 1 and i < overlapping * fps: #############################################################################  SECONDO ME E' >= 
                    # Use frames from the overlapping queue if it's not the first clip and within the overlapping range
                    frame = overlapping_queue.popleft()
                else:
                    # Read a new frame from the video
                    ret, frame = cap.read()

                    # If reading a frame failed
                    if not ret:
                        if i < clip_frames // 2: # If less than half the frames of the clip have been read
                            end_of_video = True
                            video_ended = False
                            break
                        else:
                            # Handle the case where the video ends during the latter half of the clip
                            video_length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
                            cv2.VideoCapture.set(cap, cv2.CAP_PROP_POS_FRAMES, video_length - clip_frames + i)
                            ret, frame = cap.read()

                            if not ret: # If reading the frame still fails
                                end_of_video = True
                                break
                    # Reset the end_of_video flag
                    end_of_video = False 

                # Add the frame to the overlapping queue if it's within the overlapping range
                if i >= round(clip_frames - overlapping * fps):
                    overlapping_queue.append(frame)

                # Write the current frame to the output video
                video_writer.write(frame)

                # Add label selection request in the last frame of the clip
                if i == clip_frames - 1:
                    frame = cv2.putText(frame.copy(), text="Select the label: 0, 1: ", org=(400,300), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=5)
                # Display the frame
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
                
            # Release resources if the video is ended
            if end_of_video and not video_ended:
                video_writer.release()
                os.remove(output_video_path)
                break
            
            # Wait for user input to determine the next action
            key = cv2.waitKey(0)

            # Release the video writer object
            video_writer.release()


            # Action to be performed based on the key entered
            if key == ord('n'):
                # If 'n' key is pressed, delete the current video and break the loop
                os.remove(output_video_path)
                exit_script=False
                break
            if key == ord('q'):
                # If 'q' key is pressed, delete the current video and exit the script
                os.remove(output_video_path)
                exit_script=True
                break
            if key == ord('0'):
                # If '0' key is pressed, label the current video as 0
                labels.append((output_video_relative_path, 0))
                exit_script=False
            if key == ord('1'):
                # If '1' key is pressed, label the current video as 1
                labels.append((output_video_relative_path, 1))
                exit_script=False

            # Increment the clip ID for the next clip    
            clip_id += 1
            
            # Break the loop if exit_script flag is set
            if exit_script:
                break

        # Release the video capture object
        cap.release()

        # Break the outer loop if exit_script flag is set True
        if exit_script:
            break

        # Destroy all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Destroy all OpenCV windows again as a safety measure
    cv2.destroyAllWindows()

    # Write the labels to a CSV file
    output_label_path = os.path.join(output_path, 'labels.csv')
    with open(output_label_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([('video', 'label')])
        writer.writerows(labels)

if __name__ == "__main__":
    main()