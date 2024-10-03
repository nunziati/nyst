import sys
import os
import argparse
import yaml

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.pipeline.first_pipeline import FirstPipeline
from script.flatten_video_directories import flattenVideoDirectories
from script.label_videos import labelling_videos
from script.video_rotation import process_videos_in_directory
from nyst.classifier.train import training_net, save_model_info


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

    # TRAINING FULL NET PART
    csv_input_file = yaml_configurator['csv_input_file']
    csv_label_file = yaml_configurator['csv_label_file']
    save_path = yaml_configurator['save_path']
    save_path_info = yaml_configurator['save_path_info']

    batch_size = yaml_configurator['batch_size']
    lr = yaml_configurator['lr'] 
    optimizer = yaml_configurator['optimizer']
    criterion = yaml_configurator['criterion']
    threshold_correct = yaml_configurator['threshold_correct']
    patience = yaml_configurator['patience']
    num_epochs = yaml_configurator['num_epochs'] 
    k_folds = yaml_configurator['k_folds'] 
    

    #The function returns all these variablesas a tuple, returning all the parameters as individual variables:
    return input_folder_lab, flattened_folder_lab, output_folder_lab, clip_duration, overlapping, input_folder_extr, output_folder_extr, csv_input_file, csv_label_file, save_path, save_path_info, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds


# Function to perform the pipeline and the entier code
def main(option):
    # Initialize the pipeline
    pipeline = FirstPipeline()

    # Path of the YAML configuration file
    pathConfiguratorYaml = "C:/Users/andre/OneDrive/Desktop/Altro/Tesi/code/nyst/demo/configuration.yaml"

    
    # Execute the LABELLING PHASE
    if option == 'Video labelling':
        try:
            ### YAML ###
            input_folder_lab, flattened_folder_lab, output_folder_lab, clip_duration, overlapping, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = load_hyperparams(pathConfiguratorYaml)
            
            # Flattening video directories
            flattenVideoDirectories(input_folder_lab, flattened_folder_lab)

            user_input = input('\nFlatten Step completed. Press any key to continue, otherwise press q to terminate the program: ')

            # Terminate the program
            if user_input.lower() == 'q':
                print('\nProgram terminated by the user.')
                exit()  


            rotation_input = input('\nPress y to execute rotation step: ')
            
            # Rotate the videos
            if rotation_input == 'y':
                # Usa la funzione
                process_videos_in_directory(flattened_folder_lab)
            
            user_input = input('\nRotation Step completed. Press any key to continue, otherwise press q to terminate the program: ')

            # Terminate the program
            if user_input.lower() == 'q':
                print('\nProgram terminated by the user.')
                exit()

            # Labelling videos
            labelling_videos(flattened_folder_lab, output_folder_lab, clip_duration, overlapping)

            print('\n\n --------------------------> Labelling step is COMPLETED <-------------------------- ')

        except Exception as e:
            print(f"An error occurred during the labelling phase: {e}")
            exit()

        
    # Execute the FEATURE EXTRACTION PHASE
    elif option == 'Feature Exctraction':
        try:
            ### YAML ###
            _, _, _, _, _, input_folder_extr, output_folder_extr, _, _, _, _, _, _, _, _, _, _, _, _ = load_hyperparams(pathConfiguratorYaml) 
            
            # Perform the feature extraction over all the videos in the input folder
            pipeline.videos_feature_extractor(input_folder_extr, output_folder_extr)
        
        except Exception as e:
            print(f"An error occurred during the Feature Exctraction phase: {e}")
            exit()

    
    # Execute the TRAINING AND VALIDATION PHASE
    elif option == 'Training Phase':
        try:
            ### YAML ###
            _, _, _, _, _, _, _, csv_input_file, csv_label_file, save_path, save_path_info, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds = load_hyperparams(pathConfiguratorYaml) 
            
            # Perform the training and validation of the full net using k-cross validation and grid search
            results = training_net(csv_input_file, csv_label_file, save_path, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds)

            #Save model info
            save_model_info(results, save_path_info)

        except Exception as e:
            print(f"An error occurred during the Training and Validation phase: {e}")
            exit()    
    
    elif option == 'Inference Phase':
        # Code block for option 3
        # Insert code for option 3 here
        pass
    else:
        print("Invalid option choosed.")

    

if __name__ == "__main__":

    # Select the desired option
    option = input('\nPlease select an option:\n\t1. Video labelling\n\t2. Feature Exctraction\n\t3. Training Phase\n\t4. Inference Phase\n\nYour choice: ')
    
    # Execute the main function with the selected option
    main(option)