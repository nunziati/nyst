import os
import shutil

# Specifica i percorsi delle due directory
directory1 = "/repo/porri/Eyes Segmentation Dataset/Images_f"
directory2 = "/repo/porri/Eyes Segmentation Dataset/Masks_g_f"

# Funzione per ottenere i nomi dei file senza estensione
def get_filenames_without_extension(directory):
    return {os.path.splitext(file)[0] for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))}

# Ottieni i nomi dei file senza estensione dalle due directory
names_in_directory2 = get_filenames_without_extension(directory2)

# Itera sui file nella prima directory
for file in os.listdir(directory1):
    file_path = os.path.join(directory1, file)
    
    # Controlla se il file è un'immagine
    if os.path.isfile(file_path) and file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        # Estrai il nome del file senza estensione
        file_name_without_extension = os.path.splitext(file)[0]

        # Se il nome del file non è presente nella seconda directory, rimuovilo
        if file_name_without_extension not in names_in_directory2:
            print(f"Rimuovo: {file_path}")
            os.remove(file_path)
