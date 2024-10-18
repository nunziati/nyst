import yaml

### YAML ###
def yamlParser(path_yaml_file: str):  #takes in input the yaml file path
    """
    Function required for reading a YAML file and saving its content into a variable. The input will be passed via the terminal and will be the path
    of the .yaml file.
    """
    
    with open(path_yaml_file, "r") as stream: # viene aperto il file .yaml in modalità lettura e chiamato "stream"
        yaml_parser = yaml.safe_load(stream) # salva il contenuto del file in una variabile
    return yaml_parser

def load_hyperparams(pathConfiguratorYaml: str):
    """
    This function simplifies the access to the model hyperparameters values in the YAML file. In practice, the function loads and returns the 
    hyperparameters specified in the YAML file. The function has only one input: the string of the absolute path of the file YAML.
    """
    
    yaml_configurator = yamlParser(pathConfiguratorYaml) #takes in input the path of the YAML file. It returns a Namespace containing the key-value pairs
                                                          #corresponding to the parameters specified in the file

    #Each variable takes the value corresponding to the field specified in the Namespace:

    # LABELLING PART
    input_folder_lab = yaml_configurator['input_folder_lab']
    flattened_folder_lab = yaml_configurator['flattened_folder_lab']
    output_folder_lab = yaml_configurator['output_folder_lab']
    
    clip_duration = yaml_configurator['clip_duration']
    overlapping = yaml_configurator['overlapping'] 

    # VIDEO FEATURES EXTRACTION PART   
    input_folder_extr = yaml_configurator['input_folder_extr']
    output_folder_extr = yaml_configurator['output_folder_extr']

    
    # PREPROCESS AND AUGMENTATION PART
    csv_input_file = yaml_configurator['csv_input_file']
    csv_label_file = yaml_configurator['csv_label_file']
    new_csv_file = yaml_configurator['new_csv_file']
    preprocess = yaml_configurator['preprocess']
    augmentation = yaml_configurator['augmentation']
    
    # TRAINING FULL NET PART
    save_path = yaml_configurator['save_path']
    save_path_info = yaml_configurator['save_path_info']
    save_path_wb = yaml_configurator['save_path_wb']

    

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return input_folder_lab, flattened_folder_lab, output_folder_lab, clip_duration, overlapping, input_folder_extr, output_folder_extr, csv_input_file, csv_label_file, new_csv_file, preprocess, augmentation, save_path, save_path_info, save_path_wb

# Path of the YAML configuration file
pathConfiguratorYaml = "C:/Users/andre/OneDrive/Desktop/Altro/Tesi/code/nyst/demo/configuration.yaml"
