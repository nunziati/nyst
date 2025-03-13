import sys
import os

# Add the directory 'code' to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.yaml_function import load_hyperparams, pathConfiguratorYaml


# Function to perform the pipeline and the entier code
def main(option):
        
    # Execute the LABELLING PHASE
    if option == '1':

        from script.flatten_video_directories import flattenVideoDirectories
        from script.label_videos import labelling_videos
        from script.video_rotation import process_videos_in_directory

        try:
            ### YAML ###
            input_folder_lab, flattened_folder_lab, output_folder_lab, clip_duration, overlapping, _, _, _, _, _, _, _, _, _, _ = load_hyperparams(pathConfiguratorYaml)
            
            ### FLATTENING STEP ###
            flattenVideoDirectories(input_folder_lab, flattened_folder_lab)

            user_input = input('\nFlatten Step completed. Press any key to continue, otherwise press q to terminate the program: ')
            # Terminate the program
            if user_input.lower() == 'q':
                print('\nProgram terminated by the user.')
                exit()  

            
            ### ROTATION STEP ###
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

            ### LABELLING VIDEOS STEP ###
            labelling_videos(flattened_folder_lab, output_folder_lab, clip_duration, overlapping)

            print('\n\n --------------------------> Labelling step is COMPLETED <-------------------------- ')

        except Exception as e:
            print(f"An error occurred during the labelling phase: {e}")
            exit()
        
    # Execute the FEATURE EXTRACTION PHASE
    elif option == '2':

        from nyst.pipeline.first_pipeline import FirstPipeline

        try:
            ### YAML ###
            _, _, _, _, _, input_folder_extr, output_folder_extr, _, _, _,_, _, _, _, _ = load_hyperparams(pathConfiguratorYaml) 
            
            # Initialize the pipeline
            pipeline = FirstPipeline()

            # Perform the feature extraction over all the videos in the input folder
            pipeline.videos_feature_extractor(input_folder_extr, output_folder_extr)
        
        except Exception as e:
            print(f"An error occurred during the Feature Exctraction phase: {e}")
            exit()

    # Execute the PREPROCESSING AND AUGMENTATION PHASE
    elif option == '3':
        
        import pandas as pd
        from nyst.dataset.preprocess_function import preprocess_interpolation, cubic_interpolation  
        from nyst.dataset.signal_augmentation import augment_data
        from nyst.dataset.utils_function import save_csv

        ### YAML ###
        _, _, _, _, _, _, _, csv_input_file, _, new_csv_file, preprocess, _, _, _, _ = load_hyperparams(pathConfiguratorYaml) 

        print('Loading a Custom Dataset...')
        
        # Load the CSV file
        input_data = pd.read_csv(csv_input_file)
        
        # Replace backslash with slash in both dataframes
        input_data['video'] = input_data['video'].str.replace('\\', '/')
   

        # PREPROCESSING STEP
        for prep in preprocess:
            # Preprocess signals
            if prep == 'cubic_interpolation':
                data = cubic_interpolation(input_data,150)
            elif prep == 'preprocess_interpolation':
                data = preprocess_interpolation(input_data)
            else:
                raise ValueError('Invalid preprocessing choise')
            print(f'\t ---> Preprocessing {prep} step COMPLETED\n')
        
        # Save the merged CSV
        save_csv(data, new_csv_file)
        print(f"Merged data saved to {new_csv_file}")
       
    
    # Execute the TRAINING AND VALIDATION PHASE
    elif option == '4':

        import wandb
        import yaml
        from nyst.training.train_wb import train

        try:
            # Logs into the Weights and Biases (W&B) platform, ensuring the user is authenticated
            wandb.login()

            # Load the sweep configuration from a YAML file
            with open('wb_config.yaml') as file:
                sweep_config = yaml.safe_load(file)

            # Create a sweep in W&B using the configuration loaded
            sweep_id = wandb.sweep(sweep_config, project="nyst_detection")

            # Launches the W&B agent to run the sweep
            wandb.agent(sweep_id, train, count=100)
        except Exception as e:
            print(f"An error occurred during the Training and Validation phase: {e}")
            exit()    
    
    # Execute the INFERENCE PHASE
    elif option == '5':
        # Code block for option 3
        # Insert code for option 3 here
        pass
    
    else:
        print("Invalid option choosed.")

    

if __name__ == "__main__":

    # Select the desired option
    option = input('\nPlease select an option:\n\t1. Video labelling\n\t2. Feature Exctraction\n\t3. Preprocessing\n\t4. Training Phase\n\t5. Inference Phase\n\nYour choice: ')
    
    # Execute the main function with the selected option
    main(option)