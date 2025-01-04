import os
import cv2
from ultralytics import YOLO

# Carica il modello YOLOv8 finetunato
model = YOLO('D:/model_yolo/best_yolo11m.pt')  # Sostituisci con il percorso corretto al tuo file .pt

# Specifica la cartella dei video di input e quella di output
input_folder = "D:/prova"
output_folder = "D:/prova_annoted"  # Assicurati che questa cartella esista

# Crea la cartella di output se non esiste
os.makedirs(output_folder, exist_ok=True)

# Cicla attraverso tutti i file nella cartella di input
for video_file in os.listdir(input_folder):
    if video_file.endswith('.mp4'):  # Filtra solo i file .mp4
        video_path = os.path.join(input_folder, video_file)  # Percorso completo del video
        output_path = os.path.join(output_folder, video_file[:-4] + '_annotated.mp4')  # Percorso di output

        # Esegui inferenza sul video
        results = model.predict(source=video_path, conf=0.5, iou=0.45, save=False, stream=True)

        # Inizializza il VideoWriter per salvare il video annotato
        first_frame = True  # Flag per il primo frame
        frame_count = 0  # Contatore dei frame elaborati

        for result in results:
            # Usa il metodo plot() per ottenere l'immagine con le annotazioni
            img = result.plot()  # Restituisce l'immagine con le predizioni sovrapposte

            # Verifica se l'immagine è già in formato BGR
            if img.shape[2] == 3:  # Assicurati che l'immagine abbia 3 canali
                img_bgr = img  # Utilizza direttamente l'immagine
            else:
                raise ValueError("L'immagine non ha il formato atteso con 3 canali.")

            # Ottieni le dimensioni dell'immagine per il VideoWriter
            height, width, _ = img_bgr.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il video mp4

            # Inizializza VideoWriter solo al primo frame
            if first_frame:
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                first_frame = False  # Imposta il flag su False dopo la prima inizializzazione

            # Salva il frame annotato
            out.write(img_bgr)
            frame_count += 1  # Incrementa il contatore dei frame

            # Calcola e mostra la percentuale di completamento
            # Potremmo stimare il numero totale di frame, se necessario
            percent_complete = (frame_count / 1000) * 100  # Stima con un numero arbitrario di frame
            print(f"Processing {video_file}: {percent_complete:.2f}% complete", end='\r')

        # Rilascia il VideoWriter dopo aver finito di scrivere
        out.release()

print("\nProcessing complete!")
