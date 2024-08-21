import os
import csv

# Percorso alla cartella dei video
video_dir = 'D:/nyst_labelled_videos/videos'

# Percorso al file CSV
csv_file_path = 'D:/nyst_labelled_videos/labels.csv'

# Estrai i nomi dei file video dalla cartella
video_files = set(os.listdir(video_dir))

# Estrai i nomi dei file video dal CSV
video_files_in_csv = set()

with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    # Salta l'intestazione
    next(reader)
    for row in reader:
        video_files_in_csv.add(os.path.basename(row[0]))

# Trova i file video non presenti nel CSV
videos_not_in_csv = video_files - video_files_in_csv

# Stampa i risultati
if videos_not_in_csv:
    print("I seguenti video non sono presenti nel file CSV:")
    for video in videos_not_in_csv:
        print(video)
else:
    print("Tutti i video sono presenti nel file CSV.")
