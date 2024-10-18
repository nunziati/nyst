import pandas as pd
import numpy as np
import os
import sys

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.dataset.utils_function import *

# Funzione per augmentare i dati
def augment_data(data, csv_file):
    """
    Augments the dataset by flipping the positions and speeds and appends the augmented data
    to the same CSV file.
    """
    augmented_rows = []

    # Ciclo per ogni riga nei dati
    for _, row in data.iterrows():
        try:
            # Sostituisce gli spazi nelle liste con delle virgole
            row['left_position X'] = replace_spaces_with_commas(row['left_position X'])
            row['left_position Y'] = replace_spaces_with_commas(row['left_position Y'])
            row['right_position X'] = replace_spaces_with_commas(row['right_position X'])
            row['right_position Y'] = replace_spaces_with_commas(row['right_position Y'])
            row['left_speed X'] = replace_spaces_with_commas(row['left_speed X'])
            row['left_speed Y'] = replace_spaces_with_commas(row['left_speed Y'])
            row['right_speed X'] = replace_spaces_with_commas(row['right_speed X'])
            row['right_speed Y'] = replace_spaces_with_commas(row['right_speed Y'])

            # Pulisce le virgole multiple
            row['left_position X'] = clean_comma_issues(row['left_position X'])
            row['left_position Y'] = clean_comma_issues(row['left_position Y'])
            row['right_position X'] = clean_comma_issues(row['right_position X'])
            row['right_position Y'] = clean_comma_issues(row['right_position Y'])
            row['left_speed X'] = clean_comma_issues(row['left_speed X'])
            row['left_speed Y'] = clean_comma_issues(row['left_speed Y'])
            row['right_speed X'] = clean_comma_issues(row['right_speed X'])
            row['right_speed Y'] = clean_comma_issues(row['right_speed Y'])

            # Converti le stringhe in liste di float
            left_pos_x = -np.array(parse_float_list(row['left_position X']))
            left_pos_y = -np.array(parse_float_list(row['left_position Y']))
            right_pos_x = -np.array(parse_float_list(row['right_position X']))
            right_pos_y = -np.array(parse_float_list(row['right_position Y']))
            left_speed_x = -np.array(parse_float_list(row['left_speed X']))
            left_speed_y = -np.array(parse_float_list(row['left_speed Y']))
            right_speed_x = -np.array(parse_float_list(row['right_speed X']))
            right_speed_y = -np.array(parse_float_list(row['right_speed Y']))

            # Modifica il nome del video aggiungendo '-r' prima dell'estensione
            video_path = row['video']
            base_name, ext = os.path.splitext(video_path)
            flipped_video_name = f"{base_name}-r{ext}"

            # Crea una nuova riga con i segnali invertiti
            flipped_row = {
                'video': flipped_video_name,
                'resolution': row['resolution'],
                'left_position X': left_pos_x.tolist(),
                'left_position Y': left_pos_y.tolist(),
                'right_position X': right_pos_x.tolist(),
                'right_position Y': right_pos_y.tolist(),
                'left_speed X': left_speed_x.tolist(),
                'left_speed Y': left_speed_y.tolist(),
                'right_speed X': right_speed_x.tolist(),
                'right_speed Y': right_speed_y.tolist(),
                'label': row['label']
            }

            # Aggiunge la riga invertita alla lista
            augmented_rows.append(flipped_row)

        except (ValueError, SyntaxError) as e:
            print(f"Error processing row {row['video']}: {e}")
            continue  # Salta la riga se c'è un problema con i dati

    # Converti la lista di righe augmentate in un DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Aggiungi le righe augmentate al file CSV esistente
    augmented_df.to_csv(csv_file, mode='a', header=False, index=False)
    


        
