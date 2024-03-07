import pickle

from nyst import FirstPipeline

def main():
    pipeline = FirstPipeline()
    output_dict = pipeline.run("video.mp4")

    with open("output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)
        
if __name__ == "__main__":
    main()