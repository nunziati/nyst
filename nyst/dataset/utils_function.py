import numpy as np
import re
import json
import pandas as pd


# Function to replace spaces within lists with commas
def replace_spaces_with_commas(value):
    """
    Replaces spaces within lists represented as strings with commas.

    Args:
        value (str): The value to modify, containing lists as strings. If the value is not a string, 
                     the function returns it unchanged.

    Returns:
        str: The modified string with spaces inside lists replaced by commas.
    """
    if isinstance(value, str):
        # Search inside square brackets and replace spaces with commas
        value = re.sub(r'\[\s*([^\]]+?)\s*\]', lambda m: '[' + ','.join(m.group(1).split()) + ']', value)
    return value

# Function to fix multiple commas and restore proper formatting
def clean_comma_issues(value):
    """
    Cleans up multiple consecutive commas in a string and removes any leading or trailing commas.

    Args:
        value (str): The value to clean, typically a string containing multiple commas.

    Returns:
        str: The cleaned string with consecutive commas reduced to a single comma and no leading/trailing commas.
        If the input is not a string, the function returns the original value unchanged.
    """
    if isinstance(value, str):
        # Use regex to remove consecutive commas (e.g., ,,) and replace them with a single comma
        clean_value = re.sub(r',+', ',', value)
        
        # Remove any commas at the beginning or end
        clean_value = clean_value.strip(',')
        
        # Return the cleaned value
        return clean_value
    
    return value

# Converts a string representation of a list into a Python list of floats
def parse_float_list(string_value):
    """
    Converts a string representation of a list into a Python list of floats, handling 'nan' values.

    Args:
        string_value (str or np.ndarray): A string representing a list of numerical values, or a NumPy array, which 
                                          may include 'nan' as a placeholder for missing values.

    Returns:
        list: A list of floats where 'nan' strings in the input are replaced with Python's float('nan') to represent 
              missing values. If parsing fails, an empty list is returned.
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
        
    elif isinstance(string_value, np.ndarray):  # Check if it's a np.ndarray      
        # Convert the NumPy array to a list and then to a string
        string_value = str(string_value.tolist()).replace('nan', 'null')  # Replace 'nan' with 'null'
        # Use json.loads to convert the string to a Python list
        float_list = json.loads(string_value)
        return [float('nan') if x is None else x for x in float_list]
    else:
        print(f"Warning: Expected string but got {type(string_value)}. Returning empty list.")
        return []
    
# Function to save data into a csv
def save_csv(data, csv_file):
    """
    Processes and saves the data into a CSV file. The function replaces spaces in list representations with commas,
    cleans up multiple commas, converts strings into lists of floats, and saves the modified data.

    Args:
        data (pandas.DataFrame): The dataset containing the signal data to be processed.
        csv_file (str): The path to the CSV file where the processed data will be saved.

    Returns:
        None: The function saves the processed data to a CSV file without returning anything.
    """
    new_dataset = []
    
    # Loop through each row in the data
    for _, row in data.iterrows():
        try:
            # Replace spaces in lists with commas
            row['left_position X'] = replace_spaces_with_commas(row['left_position X'])
            row['left_position Y'] = replace_spaces_with_commas(row['left_position Y'])
            row['right_position X'] = replace_spaces_with_commas(row['right_position X'])
            row['right_position Y'] = replace_spaces_with_commas(row['right_position Y'])
            row['left_speed X'] = replace_spaces_with_commas(row['left_speed X'])
            row['left_speed Y'] = replace_spaces_with_commas(row['left_speed Y'])
            row['right_speed X'] = replace_spaces_with_commas(row['right_speed X'])
            row['right_speed Y'] = replace_spaces_with_commas(row['right_speed Y'])

            # Replace multiple commas
            row['left_position X'] = clean_comma_issues(row['left_position X'])
            row['left_position Y'] = clean_comma_issues(row['left_position Y'])
            row['right_position X'] = clean_comma_issues(row['right_position X'])
            row['right_position Y'] = clean_comma_issues(row['right_position Y'])
            row['left_speed X'] = clean_comma_issues(row['left_speed X'])
            row['left_speed Y'] = clean_comma_issues(row['left_speed Y'])
            row['right_speed X'] = clean_comma_issues(row['right_speed X'])
            row['right_speed Y'] = clean_comma_issues(row['right_speed Y'])

            # Convert strings to Python lists of floats
            left_pos_x = np.array(parse_float_list(row['left_position X']))
            left_pos_y = np.array(parse_float_list(row['left_position Y']))
            right_pos_x = np.array(parse_float_list(row['right_position X']))
            right_pos_y = np.array(parse_float_list(row['right_position Y']))
            left_speed_x = np.array(parse_float_list(row['left_speed X']))
            left_speed_y = np.array(parse_float_list(row['left_speed Y']))
            right_speed_x = np.array(parse_float_list(row['right_speed X']))
            right_speed_y = np.array(parse_float_list(row['right_speed Y']))

            # Create a new row with the processed signals
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

            # Append the processed row to the list 
            new_dataset.append(flipped_row)

        except (ValueError, SyntaxError) as e:
            print(f"Error processing row {row['video']}: {e}")
            continue 

    # Convert the list of processed rows into a DataFrame
    new_dataset_df = pd.DataFrame(new_dataset)

    # Save the processed data to the specified CSV file
    new_dataset_df.to_csv(csv_file, mode='w', index=False)
