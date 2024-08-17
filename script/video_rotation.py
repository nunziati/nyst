import cv2
import os
import matplotlib.pyplot as plt

# Rotate a single frame in the specified direction
def rotate_frame(frame, direction):
    if direction == 'left':
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == 'right':
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif direction == 'down':
        return cv2.rotate(frame, cv2.ROTATE_180)
    else:
        return frame

# Rotate an entire video in the specified direction
def rotate_video(video_path, direction):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return  # Exit the function if the video cannot be opened

    # Define the codec and create a VideoWriter object to save the rotated video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 file format
    fps = cap.get(cv2.CAP_PROP_FPS)  # Retrieve the frames per second of the original video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the original video frames
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the original video frames

    # Adjust the output dimensions based on the rotation direction
    if direction in ['left', 'right']:
        # For 90-degree rotations, swap the width and height
        output_size = (frame_height, frame_width)
    else:
        # For 180-degree rotations, the dimensions remain the same
        output_size = (frame_width, frame_height)

    # Create a temporary output file to save the rotated video
    temp_output = "temp_video.mp4"
    out = cv2.VideoWriter(temp_output, fourcc, fps, output_size)

    # Loop through each frame in the original video
    while True:
        ret, frame = cap.read()  # Read a frame from the video
        if not ret:
            break  # If no more frames are available, exit the loop

        # Rotate the frame according to the specified direction
        rotated_frame = rotate_frame(frame, direction)
        
        # Write the rotated frame to the output video file
        out.write(rotated_frame)

    # Release the resources associated with video reading and writing
    cap.release()
    out.release()

    # Replace the original video with the rotated video by renaming the temp file
    os.replace(temp_output, video_path)
    print(f"The video {video_path} has been rotated {direction}.")

# Display the first frame of the video and then rotate the video based on user input
def show_first_frame_and_rotate(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return  # Exit the function if the video cannot be opened
    
    # Read the first frame of the video
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Cannot read the first frame of the video {video_path}")
        return  # Exit the function if the first frame cannot be read
    
    # Display the first frame to the user
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the axis for better visualization
    plt.show()  # Display the frame

    # Ask the user to input the direction in which they want to rotate the video
    direction = input("Enter the rotation direction (left, right, or down): ").strip().lower()

    # Validate the user input to ensure it's a valid direction
    if direction not in ['left', 'right', 'down']:
        print("Invalid direction. Please enter 'left', 'right', or 'down'.")
        return  # Exit the function if the input is invalid
    
    # If the input is valid, proceed to rotate the video in the specified direction
    rotate_video(video_path, direction)

# Process all videos in the specified directory
def process_videos_in_directory(directory_path):
    # Loop through all videos in the given directory
    for filename in os.listdir(directory_path):
        # Check if the file is a video by its extension
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # Construct the full path to the video file
            video_path = os.path.join(directory_path, filename)
            print(f"Processing video: {filename}")  # Inform the user which video is being processed
            
            # Call the function to display the first frame and allow the user to rotate the video
            show_first_frame_and_rotate(video_path)
