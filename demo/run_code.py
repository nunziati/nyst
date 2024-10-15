import sys
import os

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.yaml_function import load_hyperparams, pathConfiguratorYaml


# Function to perform the pipeline and the entier code
def main(option):
        
    # Execute the LABELLING PHASE
    if option == 'Video labelling':

        from script.flatten_video_directories import flattenVideoDirectories
        from script.label_videos import labelling_videos
        from script.video_rotation import process_videos_in_directory

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

        from nyst.pipeline.first_pipeline import FirstPipeline

        try:
            ### YAML ###
            _, _, _, _, _, input_folder_extr, output_folder_extr, _, _, _, _, _, _, _, _, _, _, _, _ = load_hyperparams(pathConfiguratorYaml) 
            
            # Initialize the pipeline
            pipeline = FirstPipeline()

            # Perform the feature extraction over all the videos in the input folder
            pipeline.videos_feature_extractor(input_folder_extr, output_folder_extr)
        
        except Exception as e:
            print(f"An error occurred during the Feature Exctraction phase: {e}")
            exit()

    # Execute the TRAINING AND VALIDATION PHASE
    elif option == 'Training Phase':

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