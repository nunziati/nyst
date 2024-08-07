import pickle
import sys
import os

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.pipeline.first_pipeline import FirstPipeline

def main():
    pipeline = FirstPipeline()
    '''
    output_dict = pipeline.run('D:/nyst_video_cartelle/040/001.mp4') #D:/nyst_labelled_videos/videos/001_001_001.mp4
    
    with open("output_dict.pkl", "wb") as f:
         pickle.dump(output_dict, f)'''
    
    input_folder = 'D:/nyst_labelled_videos/videos'
    output_folder = 'D:/nyst_labelled_videos'
    pipeline.videos_feature_extractor(input_folder, output_folder)

if __name__ == "__main__":
    main()