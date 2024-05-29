import pickle
import cv2

from nyst.pipeline import FirstPipeline

def main():
    pipeline = FirstPipeline()
    output_dict = pipeline.run('/repo/porri/20.mp4')
    
    with open("output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)


if __name__ == "__main__":
    main()