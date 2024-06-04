import os
import pickle
from collections import deque
import csv
import argparse

import cv2

video_extensions = ['.mp4', '.mkv']

def parse_args():
    parser = argparse.ArgumentParser(description='Label videos')
    parser.add_argument('--resume', action='store_true', help='Resume labeling')

    return parser.parse_args()

# Settings
root = '/repo/porri/dataset_video'
output_path = '/repo/porri/output_videos'
clip_duration = 10 # in seconds
overlapping = 8 # in seconds

def main():
    args = parse_args()

    labels = [] 

    os.makedirs(output_path, exist_ok=True)

    videos = os.listdir(root)
    videos = sorted([video for video in videos if any(video.endswith(ext) for ext in video_extensions)])

    if args.resume:
        with open('labels.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            labels = list(reader)
    
    for video_idx, video in enumerate(videos):
        cv2.namedWindow('Frame')
        cv2.resizeWindow('Frame', 800, 600)
        
        if args.resume and video_idx < len(videos) - 1 and os.path.exists(output_path, os.path.splitext(videos[video_idx + 1])[0]):
            continue
        
        video_path = os.path.join(root, video)
        output_video_common_path = os.path.join(output_path, os.path.splitext(video)[0])
        
        os.makedirs(output_video_common_path, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise("Error opening video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        clip_frames = round(clip_duration * fps)

        overlapping_queue = deque()
        clip_id = 0

        while True:
            if not args.resume or os.path.join(output_video_common_path, f'{clip_id}.mp4') not in [label[0] for label in labels]:
                video_writer = cv2.VideoWriter(os.path.join(output_video_common_path, f'{clip_id}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(cap.get(3)), int(cap.get(4))))
            
            for i in range(clip_frames):
                if clip_id != 0 and i < overlapping * fps:
                    frame = overlapping_queue.popleft()
                else:
                    ret, frame = cap.read()

                    if not ret:
                        end_of_video = True
                        break
                    end_of_video = False

                if i >= round(clip_frames - overlapping * fps):
                    overlapping_queue.append(frame)

                if args.resume and os.path.join(output_video_common_path, f'{clip_id}.mp4') in [label[0] for label in labels]:
                    continue

                video_writer.write(frame)

                if i == clip_frames - 1:
                    frame = cv2.putText(frame, text="Select the label: 0, 1", org=(400,300), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,0,0), thickness=5)
                cv2.imshow('Frame', frame)
                cv2.waitKey(10)
            if end_of_video:
                video_writer.release()
                os.remove(os.path.join(output_video_common_path, f'{clip_id}.mp4'))
                break

            if not args.resume or os.path.join(output_video_common_path, f'{clip_id}.mp4') not in [label[0] for label in labels]:
                key = cv2.waitKey(0)

                video_writer.release()

                if key == ord('n'):
                    os.remove(os.path.join(output_video_common_path, f'{clip_id}.mp4'))
                    exit_script=False
                    break
                if key == ord('q'):
                    os.remove(os.path.join(output_video_common_path, f'{clip_id}.mp4'))
                    exit_script=True
                    break
                if key == ord('0'):
                    labels.append((os.path.join(output_video_common_path, f'{clip_id}.mp4'), 0))
                    exit_script=False
                if key == ord('1'):
                    labels.append((os.path.join(output_video_common_path, f'{clip_id}.mp4'), 1))
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
        writer.writerows(labels)

if __name__ == "__main__":
    main()