import os
import cv2
import pandas as pd
import sys
import numpy as np

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.dataset import CustomDataset

# Funzione per ottenere il framerate dei video
def get_video_framerate(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()
    return fps

# Funzione per elencare il numero di video per ogni framerate e i nomi dei video
def count_videos_by_framerate(video_folder):
    framerate_data = {}
    
    for filename in os.listdir(video_folder):
        if filename.endswith((".mp4", ".avi", ".mov", ".mkv")):  # Aggiungi i formati video rilevanti
            video_path = os.path.join(video_folder, filename)
            try:
                framerate = get_video_framerate(video_path)
                if framerate not in framerate_data:
                    framerate_data[framerate] = {
                        'count': 0,
                        'videos': []
                    }
                framerate_data[framerate]['count'] += 1
                framerate_data[framerate]['videos'].append(filename)
            except Exception as e:
                print(f"Errore con il file {filename}: {e}")
    
    return framerate_data

# Funzione per leggere il CSV e contare i video per classe
def count_videos_by_class(csv_path):
    df = pd.read_csv(csv_path)
    
    # Controlla se ci sono le colonne corrette
    if 'video' not in df.columns or 'label' not in df.columns:
        raise ValueError("Il CSV deve contenere le colonne 'video' e 'label'")
    
    # Conteggio dei video per classe
    class_counts = df['label'].value_counts().to_dict()
    
    # Lista dei video con la rispettiva etichetta
    video_label_list = df[['video', 'label']].values.tolist()
    
    return class_counts, video_label_list


def count_data_before_after_filter(original_data, filtered_data):
    """
    Conta il numero di video e i video per classe prima e dopo il filtraggio.

    Arguments:
    - original_data (pd.DataFrame): Il dataframe contenente i dati originali prima del filtraggio.
    - filtered_data (dict): Un dizionario contenente i dati filtrati, inclusi i labels.

    Returns:
    - dict: Un dizionario che contiene il numero totale di video e il conteggio dei video per classe,
            sia prima che dopo il filtraggio.
    """
    # Conteggio totale dei video pre-filtraggio
    total_videos_pre_filter = len(original_data)

    # Conteggio dei video per classe pre-filtraggio
    class_counts_pre_filter = original_data['label'].value_counts().to_dict()

    # Conteggio totale dei video post-filtraggio
    total_videos_post_filter = len(filtered_data['labels'])

    # Conteggio dei video per classe post-filtraggio
    labels_post_filter = filtered_data['labels'].flatten()  # Trasformiamo l'array in una lista piatta
    unique, counts = np.unique(labels_post_filter, return_counts=True)
    class_counts_post_filter = dict(zip(unique, counts))

    return {
        'pre_filter': {
            'total_videos': total_videos_pre_filter,
            'class_counts': class_counts_pre_filter
        },
        'post_filter': {
            'total_videos': total_videos_post_filter,
            'class_counts': class_counts_post_filter
        }
    }


# Funzione per contare il numero di video invalidi per ciascun motivo
def count_invalid_videos_by_reason(invalid_video_info):
    '''
    Conta il numero di video invalidi per ciascun motivo di filtraggio.

    Arguments:
    - invalid_video_info (list): Una lista contenente informazioni sui video scartati e i motivi per cui sono stati scartati.

    Returns:
    - dict: Un dizionario che associa ogni motivo al numero di video scartati.
    '''
    reason_counts = {}
    
    for video_info in invalid_video_info:
        reason = video_info['reason']
        if reason not in reason_counts:
            reason_counts[reason] = 0
        reason_counts[reason] += 1
    
    return reason_counts

# Funzione per scrivere i risultati su un file di testo
def write_results_to_file(framerate_data, class_counts, video_label_list, invalid_video_info, invalid_reason_counts, output_file):
    with open(output_file, 'w') as f:
        # Scrivi i dati relativi ai framerate
        f.write("Numero di video per framerate:\n")
        for framerate, data in framerate_data.items():
            f.write(f"{data['count']} video a {framerate} fps\n")
            f.write(f"Video associati a {framerate} fps:\n")
            for video in data['videos']:
                f.write(f"    {video}\n")
        f.write("\n")
        
    

        # Scrivi le informazioni sui video filtrati
        f.write("\nInformazioni sui segnali dei video filtrati e successivamente eliminati:\n")
        if invalid_video_info:
            for video_info in invalid_video_info:
                f.write(f"Video: {video_info['video']}, Motivo: {video_info['reason']}\n")
        else:
            f.write("Nessun segnale video filtrato ed eliminato.\n")

        # Scrivi il conteggio dei video filtrati per motivo
        f.write("\nConteggio dei segnali dei video eliminati e suddivisi per motivo eliminazione:\n")
        for reason, count in invalid_reason_counts.items():
            f.write(f"Motivo: {reason} -----> Numero di segnali video: {count}\n")

        # Scrivi i dati relativi alle classi
        f.write("\nConteggio dei segnali ottenuti dai video, suddivisi per classe, pre-filtraggio:\n")
        for label, count in filtering_report['pre_filter']['class_counts'].items():
            f.write(f"Classe {label}: {count} video\n")

        f.write("\nConteggio dei segnali ottenuti dai video, suddivisi per classe, post-filtraggio:\n")
        for label, count in filtering_report['post_filter']['class_counts'].items():
            f.write(f"Classe {label}: {count} video\n")

        f.write(f"\nNumero totale di video pre-filtraggio: {filtering_report['pre_filter']['total_videos']}\n")
        f.write(f"Numero totale di video post-filtraggio: {filtering_report['post_filter']['total_videos']}\n")
        
        

if __name__ == '__main__':
    # Path specificati dall'utente
    video_folder_path = "D:/nyst_labelled_videos/videos"
    csv_path_lab = "D:/nyst_labelled_videos/labels.csv"
    csv_path_inp = "D:/nyst_labelled_videos/video_features.csv"
    output_file = "D:/nyst_labelled_videos/report.txt"

    # Chiamata alla funzione per ottenere il framerate dei video
    framerate_data = count_videos_by_framerate(video_folder_path)

    # Chiamata alla funzione per ottenere il conteggio dei video per classe e la lista dei video con etichetta
    class_counts, video_label_list = count_videos_by_class(csv_path_lab)

    # Ottenere i video filtrati con motivi di filtro
    dataset = CustomDataset(csv_input_file=csv_path_inp, csv_label_file=csv_path_lab)  # Assicurati che i path siano corretti
    invalid_video_info = dataset.invalid_video_info  # Video filtrati con i motivi

    # Dataset originale (pre-filtraggio)
    original_data = dataset.merged_data  # DataFrame originale prima del filtraggio

    # Dataset filtrato (post-filtraggio)
    filtered_data = dataset.data  # Dizionario di dati dopo il filtraggio

    # Conta i dati prima e dopo il filtraggio
    filtering_report = count_data_before_after_filter(original_data, filtered_data)


    # Contare il numero di video scartati per ogni motivo
    invalid_reason_counts = count_invalid_videos_by_reason(invalid_video_info)
    
    # Scrivi i risultati su un file di testo
    write_results_to_file(framerate_data, class_counts, video_label_list, invalid_video_info, invalid_reason_counts, output_file)

    print(f"I risultati sono stati scritti su {output_file}")
