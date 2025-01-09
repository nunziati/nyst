import cv2
import os
import numpy as np

# Funzione per convertire un'immagine in scala di grigi e rimappare i livelli di grigio
def convert_and_remap_gray(img):
    # Converti l'immagine in scala di grigi
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    '''cv2.imshow("gray",gray_img)
    cv2.waitKey(100000)'''
    unique_gray_levels = np.unique(gray_img)
    
    # Rimappa i livelli di grigio
    remapped_gray_img = np.zeros_like(gray_img)
    mapped_class = 0
    for level in unique_gray_levels:
        remapped_gray_img[gray_img == level] = mapped_class
        mapped_class += 1

    return remapped_gray_img

# Funzione per ciclare attraverso le directory e convertire le immagini
def process_images(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.png'):  # Aggiungi altri formati di immagine se necessario
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                remapped_gray_img = convert_and_remap_gray(img)

                # Crea la directory di destinazione se non esiste
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                # Salva l'immagine convertita
                output_path = os.path.join(output_subdir, file)
                cv2.imwrite(output_path, remapped_gray_img)

# Funzione per stampare i livelli di grigio presenti in un'immagine
def print_gray_levels(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Leggi l'immagine in scala di grigi
    unique_gray_levels = np.unique(img)  # Trova i livelli di grigio unici presenti nell'immagine
    print(f"Livelli di grigio presenti nell'immagine '{img_path}':")
    for level in unique_gray_levels:
        print(level)

# Definisci le directory di input e output
input_directory = '/repo/porri/Eyes Segmentation Dataset/Masks'
output_directory = '/repo/porri/Eyes Segmentation Dataset/Masks_g'

# Esegui la conversione
process_images(input_directory, output_directory)

'''
# Esempio di utilizzo
image_path =input("Path immagine: ")

if image_path != "o":
    print_gray_levels(image_path)
else:
    pass'''
