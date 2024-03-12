import pickle

from nyst.pipeline import FirstPipeline

def main():
    pipeline = FirstPipeline()
    output_dict = pipeline.run("/home/giacomo/nystagmus/data/Paziente cinese 1. orizzontale destro/20230901-133502-rotated.mp4")

    with open("output_dict.pkl", "wb") as f:
        pickle.dump(output_dict, f)

if __name__ == "__main__":
    main()