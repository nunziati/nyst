import numpy as np
import re
import json
import pandas as pd


# Funzione per sostituire gli spazi all'interno delle liste con delle virgole
def replace_spaces_with_commas(value):
    if isinstance(value, str):  # Verifica se il valore è una stringa
        # Cerca all'interno delle parentesi quadre e sostituisci gli spazi con delle virgole
        value = re.sub(r'\[\s*([^\]]+?)\s*\]', lambda m: '[' + ','.join(m.group(1).split()) + ']', value)
    return value  # Restituisce il valore modificato

# Funzione per correggere le virgole multiple e ripristinare la formattazione corretta
def clean_comma_issues(value):
    if isinstance(value, str):  # Verifica se il valore è una stringa
        # Usa una regex per rimuovere virgole consecutive (es. ,,) e sostituiscile con una sola virgola
        clean_value = re.sub(r',+', ',', value)
        
        # Rimuovi eventuali virgole all'inizio o alla fine
        clean_value = clean_value.strip(',')
        
        # Restituisci il valore pulito
        return clean_value
    return value  # Restituisce il valore così com'è se non è una stringa

def parse_float_list(string_value):
    """
    Converts a string representation of a list into a Python list of floats, handling 'nan' values.

    Arguments:
    - string_value (str): A string that represents a list of numerical values, which may include 'nan' as a placeholder for missing values.

    Returns:
    - float_list (list): A list of floats where 'nan' strings in the input are replaced with Python's float('nan') to represent missing values.
    """
    if isinstance(string_value, str):  # Check if it's a string
        string_value = string_value.replace('nan', 'null')
        try:
            # Try to use json.loads to parse the string
            float_list = json.loads(string_value)
            # Replace 'null' with float('nan')
            return [float('nan') if x is None else x for x in float_list]
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {string_value}")
            return []  # Return an empty list in case of a decoding error
        
    elif isinstance(string_value, np.ndarray):         
        # Convert the NumPy array to a list and then to a string
        string_value = str(string_value.tolist()).replace('nan', 'null')  # Replace 'nan' with 'null'
        # Use json.loads to convert the string to a Python list
        float_list = json.loads(string_value)
        return [float('nan') if x is None else x for x in float_list]
    else:
        print(f"Warning: Expected string but got {type(string_value)}. Returning empty list.")
        return []
    
# Funzione per salvare i dati
def save_csv(data, csv_file):

    new_dataset = []
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
            left_pos_x = np.array(parse_float_list(row['left_position X']))
            left_pos_y = np.array(parse_float_list(row['left_position Y']))
            right_pos_x = np.array(parse_float_list(row['right_position X']))
            right_pos_y = np.array(parse_float_list(row['right_position Y']))
            left_speed_x = np.array(parse_float_list(row['left_speed X']))
            left_speed_y = np.array(parse_float_list(row['left_speed Y']))
            right_speed_x = np.array(parse_float_list(row['right_speed X']))
            right_speed_y = np.array(parse_float_list(row['right_speed Y']))

            # Crea una nuova riga con i segnali invertiti
            flipped_row = {
                'video': row['video'],
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
            new_dataset.append(flipped_row)

        except (ValueError, SyntaxError) as e:
            print(f"Error processing row {row['video']}: {e}")
            continue  # Salta la riga se c'è un problema con i dati

    # Converti la lista di righe augmentate in un DataFrame
    new_dataset_df = pd.DataFrame(new_dataset)

    # Aggiungi le righe augmentate al file CSV esistente
    new_dataset_df.to_csv(csv_file, mode='w', index=False)
