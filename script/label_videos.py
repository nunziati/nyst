import os
from collections import deque
import csv
import cv2

# Define supported video file extensions
video_extensions = ['.mp4', '.mkv']


# Main function to handle videos labelling process
def labelling_videos(input_path, output_path, clip_duration=10, overlapping=8):

    # List to store video labels
    labels = [] 

    # Create output directory for videos if it doesn't exist
    os.makedirs(os.path.join(output_path, "videos"), exist_ok=True)

    # Get the list of ordered video names from the input directory
    videos = os.listdir(input_path)
    videos = sorted([video for video in videos if any(video.endswith(ext) for ext in video_extensions)])
    
    # Loop through the list of videos
    for video in videos:
        
        # Window to visualize the video 
        cv2.namedWindow('Frame')
        cv2.resizeWindow('Frame', 800, 600)
        
        # Create the video path
        video_path = os.path.join(input_path, video)

        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video {video}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS) # Get number of fps
        clip_frames = round(clip_duration * fps) # Number of frames per clip
        overlapping_frames = round(overlapping * fps) # Number of overlapping frames
        video_total_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames in the video

        # Calculate the number of clips based on video length
        if video_total_length <= clip_frames:
            # If video length is less than or equal to clip duration, just create one clip
            num_clips = 1
        else:
            num_clips = ((video_total_length - clip_frames) // (clip_frames - overlapping_frames)) + 1 # Delete the last frames (otherwise write +2)

        # Queue to store overlapping frames
        overlapping_queue = deque()
        clip_id = 1

        while clip_id <= num_clips:
            output_video_name = video.split('.')[0] + f'_{str(clip_id).rjust(3, "0")}.mp4' # Generate the output video file name
            output_video_relative_path = os.path.normpath(os.path.join("videos", output_video_name)) # Create the relative path for the output video
            output_video_path = os.path.join(output_path, output_video_relative_path) # Create the full path for the output video
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4)))) # Initialize the video writer object
            
            # Flags
            end_of_video = False # Flag to indicate if the end of the video has been reached

            # Loop over the frames in a clip
            for i in range(clip_frames):
                if clip_id > 1 and i < overlapping * fps:  
                    # Use frames from the overlapping queue if it's not the first clip and within the overlapping range
                    frame = overlapping_queue.popleft()
                else:
                    # Read a new frame from the video
                    ret, frame = cap.read()

                    # If reading a frame failed
                    if not ret:
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
            if end_of_video:
                video_writer.release()
                os.remove(output_video_path)
                break
            
            # Wait for user input to determine the next action
            key = cv2.waitKey(0)

            # Release the video writer object
            video_writer.release()


            # Action to be performed based on the key entered
            if key == ord('n'):
                # If 'n' key is pressed, delete the current video and continue
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
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    # Destroy all OpenCV windows again as a safety measure
    cv2.destroyAllWindows()

    # Write the labels to a CSV file
    output_label_path = os.path.join(output_path, 'labels.csv')
    file_exists = os.path.isfile(output_label_path)

    with open(output_label_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['video', 'label'])  # write header only if file does not exist
        writer.writerows(labels)

