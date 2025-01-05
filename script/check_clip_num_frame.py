import os
import cv2

def analyze_videos_in_folder(folder_path, output_file="D:/nyst_labelled_videos_prova/video_analysis.txt"):
    # Controlla se la cartella esiste
    if not os.path.exists(folder_path):
        print(f"Errore: la cartella '{folder_path}' non esiste.")
        return
    
    # Apri il file di output
    with open(output_file, "w") as f:
        f.write("Analisi dei video nella cartella:\n")
        f.write(f"Cartella: {folder_path}\n\n")
        f.write(f"{'Nome video':<30}{'Frame totali':<15}{'FPS':<10}{'Durata (s)':<15}\n")
        f.write("-" * 70 + "\n")
        
        # Scansiona i file nella cartella
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            
            # Controlla se è un file video supportato
            if os.path.isfile(file_path) and file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                try:
                    # Carica il video
                    cap = cv2.VideoCapture(file_path)
                    
                    if not cap.isOpened():
                        print(f"Impossibile aprire il video: {file_name}")
                        continue
                    
                    # Estrai informazioni
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    duration = frame_count / fps if fps > 0 else 0
                    
                    # Scrivi le informazioni sul file
                    f.write(f"{file_name:<30}{frame_count:<15}{fps:<10.2f}{duration:<15.2f}\n")
                    
                    print(f"Analizzato: {file_name}")
                    
                except Exception as e:
                    print(f"Errore durante l'analisi di '{file_name}': {e}")
                finally:
                    cap.release()
        print(f"Analisi completata. Risultati salvati in '{output_file}'.")

# Esegui lo script
folder_path = 'D:/nyst_labelled_videos_prova/videos'
analyze_videos_in_folder(folder_path)
