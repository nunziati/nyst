import pandas as pd
import numpy as np
import os
import sys

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.dataset.utils_function import *

# Function to invert signals directions to augment data 
def augment_data(data, csv_file):
    """
    Augments the provided DataFrame by flipping positional and speed data, 
    and appending the augmented data to a specified CSV file.

    Args:
        data (pd.DataFrame): A DataFrame containing the original data with columns 
                             'left_position X', 'left_position Y', 'right_position X', 
                             'right_position Y', 'left_speed X', 'left_speed Y', 
                             'right_speed X', 'right_speed Y', 'video', 'resolution', 
                             and 'label'.
        csv_file (str): The file path of the CSV file where the augmented data 
                        will be appended.

    Returns:
        None: This function does not return any value. Instead, it appends the 
              augmented data directly to the specified CSV file.
    """    
    # Function to augment data
    augmented_rows = []

    # Loop through each row in the data
    for _, row in data.iterrows():
        try:
            # Replace spaces in list-like strings with commas
            row['left_position X'] = replace_spaces_with_commas(row['left_position X'])
            row['left_position Y'] = replace_spaces_with_commas(row['left_position Y'])
            row['right_position X'] = replace_spaces_with_commas(row['right_position X'])
            row['right_position Y'] = replace_spaces_with_commas(row['right_position Y'])
            row['left_speed X'] = replace_spaces_with_commas(row['left_speed X'])
            row['left_speed Y'] = replace_spaces_with_commas(row['left_speed Y'])
            row['right_speed X'] = replace_spaces_with_commas(row['right_speed X'])
            row['right_speed Y'] = replace_spaces_with_commas(row['right_speed Y'])

            # Clean up multiple commas in the data
            row['left_position X'] = clean_comma_issues(row['left_position X'])
            row['left_position Y'] = clean_comma_issues(row['left_position Y'])
            row['right_position X'] = clean_comma_issues(row['right_position X'])
            row['right_position Y'] = clean_comma_issues(row['right_position Y'])
            row['left_speed X'] = clean_comma_issues(row['left_speed X'])
            row['left_speed Y'] = clean_comma_issues(row['left_speed Y'])
            row['right_speed X'] = clean_comma_issues(row['right_speed X'])
            row['right_speed Y'] = clean_comma_issues(row['right_speed Y'])

            # Convert cleaned strings into lists of floats and negate the values
            left_pos_x = -np.array(parse_float_list(row['left_position X']))
            left_pos_y = -np.array(parse_float_list(row['left_position Y']))
            right_pos_x = -np.array(parse_float_list(row['right_position X']))
            right_pos_y = -np.array(parse_float_list(row['right_position Y']))
            left_speed_x = -np.array(parse_float_list(row['left_speed X']))
            left_speed_y = -np.array(parse_float_list(row['left_speed Y']))
            right_speed_x = -np.array(parse_float_list(row['right_speed X']))
            right_speed_y = -np.array(parse_float_list(row['right_speed Y']))

            # Modify the video name by appending '-r' before the file extension
            video_path = row['video']
            base_name, ext = os.path.splitext(video_path)
            flipped_video_name = f"{base_name}-r{ext}"

            # Create a new row with the flipped signals
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

            # Add the flipped row to the list
            augmented_rows.append(flipped_row)

        # Handle potential data errors and continue with the next row
        except (ValueError, SyntaxError) as e:
            print(f"Error processing row {row['video']}: {e}")
            continue  # Salta la riga se c'è un problema con i dati

    # Convert the list of augmented rows into a DataFrame
    augmented_df = pd.DataFrame(augmented_rows)

    # Append the augmented data to the existing CSV file
    augmented_df.to_csv(csv_file, mode='a', header=False, index=False)
    


        
