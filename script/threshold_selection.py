import csv

# Percorsi dei file di input e output
input_csv = "C:/Users/andre/Downloads/input.csv"
output_csv = "C:/Users/andre/Downloads/output_with_diff.csv"

left_iris_listlist = []
left_pupil_listlist = []
right_iris_listlist = []
right_pupil_listlist = []
video_list = []

try:
    # Lettura del file di input CSV
    with open(input_csv, "r") as csvfile:
        reader = csv.DictReader(csvfile)  # Legge il CSV come dizionari
        
        for row in reader:
            # Parsing dei dati
            video_name = row["Video"]
            left_iris = eval(row["left_iris_list"])  # Usa eval per convertire stringhe in liste
            left_pupil = eval(row["left_pupil_list"])
            right_iris = eval(row["right_iris_list"])
            right_pupil = eval(row["right_pupil_list"])
            
            # Aggiungi alle liste
            video_list.append(video_name)
            left_iris_listlist.append(left_iris)
            left_pupil_listlist.append(left_pupil)
            right_iris_listlist.append(right_iris)
            right_pupil_listlist.append(right_pupil)
    
    # Calcolo delle differenze
    diff_left = []
    diff_right = []

    # Calcolo di diff_left
    for iris_list, pupil_list in zip(left_iris_listlist, left_pupil_listlist):
        diff = [i - p for i, p in zip(iris_list, pupil_list)]
        diff_left.append(diff)

    # Calcolo di diff_right
    for iris_list, pupil_list in zip(right_iris_listlist, right_pupil_listlist):
        diff = [i - p for i, p in zip(iris_list, pupil_list)]
        diff_right.append(diff)

    # Creazione del file di output CSV
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = ["Video", "diff_left", "diff_right"]  # Definiamo le intestazioni del CSV
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Scrittura dell'intestazione
        writer.writeheader()
        
        # Scrittura dei dati riga per riga
        for i in range(len(video_list)):
            writer.writerow({
                "Video": video_list[i],
                "diff_left": diff_left[i],
                "diff_right": diff_right[i]
            })
    
    print(f"File salvato con successo in {output_csv}.")
except FileNotFoundError:
    print(f"Il file di input {input_csv} non è stato trovato.")
except ValueError as ve:
    print(f"Errore di parsing: {ve}")
except Exception as e:
    print(f"Si è verificato un errore: {e}")
