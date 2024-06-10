import os
import pickle
from collections import deque
import csv
import argparse

import cv2

video_extensions = ['.mp4', '.mkv']

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

def main():
    # Default settings
    root = '/repo/porri/dataset_video'
    output_path = '/repo/porri/output_videos'
    clip_duration = 10 # in seconds
    overlapping = 8 # in seconds

    default_args = {'input_dir': root, 'output_dir': output_path, 'clip_duration': clip_duration, 'overlapping': overlapping}

    args = parse_args(default_args)

    root = args.input_dir
    output_path = args.output_dir
    clip_duration = args.clip_duration
    overlapping = args.overlapping

    labels = [] 

    os.makedirs(os.path.join(output_path, "videos"), exist_ok=True)

    videos = os.listdir(root)
    videos = sorted([video for video in videos if any(video.endswith(ext) for ext in video_extensions)])
    
    for video_idx, video in enumerate(videos):
        cv2.namedWindow('Frame')
        cv2.resizeWindow('Frame', 800, 600)
        
        video_path = os.path.join(root, video)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise("Error opening video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        clip_frames = round(clip_duration * fps)

        overlapping_queue = deque()
        clip_id = 1

        while True:
            output_video_name = video.split('.')[0] + f'_{str(clip_id).rjust(3, "0")}.mp4'
            output_video_path = os.path.join(output_path, "videos", output_video_name)
            video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
            
            end_of_video = False
            video_ended = True

            for i in range(clip_frames):
                if clip_id > 1 and i < overlapping * fps:
                    frame = overlapping_queue.popleft()
                else:
                    ret, frame = cap.read()

                    if not ret:
                        if i < clip_frames // 2:
                            end_of_video = True
                            video_ended = False
                            break
                        else:
                            video_length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
                            cv2.VideoCapture.set(cap, cv2.CAP_PROP_POS_FRAMES, video_length - clip_frames + i)
                            ret, frame = cap.read()

                            if not ret:
                                end_of_video = True
                                break

                    end_of_video = False

                if i >= round(clip_frames - overlapping * fps):
                    overlapping_queue.append(frame)

                video_writer.write(frame)

                if i == clip_frames - 1:
                    frame = cv2.putText(frame, text="Select the label: 0, 1", org=(400,300), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=5)
                cv2.imshow('Frame', frame)
                cv2.waitKey(20)
                
            if end_of_video and not video_ended:
                video_writer.release()
                os.remove(output_video_path)
                break

            key = cv2.waitKey(0)

            video_writer.release()

            if key == ord('n'):
                os.remove(output_video_path)
                exit_script=False
                break
            if key == ord('q'):
                os.remove(output_video_path)
                exit_script=True
                break
            if key == ord('0'):
                labels.append((output_video_path, 0))
                exit_script=False
            if key == ord('1'):
                labels.append((output_video_path, 1))
                exit_script=False
                
            clip_id += 1
            
            if exit_script:
                break

        cap.release()

        if exit_script:
            break

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

    with open('labels.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([('video', 'label')])
        writer.writerows(labels)

if __name__ == "__main__":
    main()