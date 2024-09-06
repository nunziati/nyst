import os
import cv2

def extract_video_info(input_directory, output_file):
    # Create a list to store video information
    video_info = []
    
    # Loop through each file in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4") or filename.endswith(".avi") or filename.endswith(".mov"):  # Add more extensions if needed
            video_path = os.path.join(input_directory, filename)
            
            # Open the video with OpenCV
            video = cv2.VideoCapture(video_path)
            
            if not video.isOpened():
                print(f"Unable to open video {filename}")
                continue
            
            # Extract information
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Number of frames
            fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second (FPS)
            
            # Store the information in the list
            video_info.append(f"Video Name: {filename}\nFrame Count: {frame_count}\nFPS: {fps}\n")
            
            # Release the video capture object
            video.release()
    
    # Save the information in a text file at the specified output path
    with open(output_file, "w") as file:
        for info in video_info:
            file.write(info + "\n")
    
    print(f"Video information has been saved")



if __name__ == "__main__":
    # Example usage
    directory_path = "D:/nyst_labelled_videos/videos"
    output_path = "D:/nyst_labelled_videos/videos_videoinfo.txt"
    extract_video_info(directory_path,output_path)

    # Example usage annotated
    directory_path = "D:/nyst_labelled_videos/Annotated_videos"
    output_path = "D:/nyst_labelled_videos/videos_videoinfo_annotated.txt"
    extract_video_info(directory_path,output_path)
