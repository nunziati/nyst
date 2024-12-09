import os

def find_inconsistencies(images_folder, labels_folder, output_file):
    # Ottieni la lista di file nelle directory
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if os.path.isfile(os.path.join(labels_folder, f))}
    
    # Trova immagini senza etichette e viceversa
    images_without_labels = image_files - label_files
    labels_without_images = label_files - image_files
    
    # Scrivi i risultati in un file di testo
    with open(output_file, 'w') as f:
        if images_without_labels:
            f.write("Images without corresponding labels:\n")
            for img in images_without_labels:
                f.write(f"{img}\n")
        else:
            f.write("No images without corresponding labels found.\n")
        
        f.write("\n")
        
        if labels_without_images:
            f.write("Labels without corresponding images:\n")
            for lbl in labels_without_images:
                f.write(f"{lbl}\n")
        else:
            f.write("No labels without corresponding images found.\n")
    
    print(f"Inconsistencies saved to {output_file}")

# Usa le cartelle corrette qui
images_folder = "C:/Users/andre/Downloads/dataset/train/images"
labels_folder = "C:/Users/andre/Downloads/dataset/train/labels"
output_file = "C:/Users/andre/Downloads/inconsistencies.txt"


if __name__ == "__main__":
    find_inconsistencies(images_folder, labels_folder, output_file)
