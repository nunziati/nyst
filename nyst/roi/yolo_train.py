from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
from roboflow import Roboflow
import wandb
import torch
import yaml
import shutil

# Function to train the YOLO model for Eyes Detection
def train(exp_config):
    """
    Trains a YOLOv8 model for eye detection using the provided experiment configuration.

    Args:
        exp_config (dict): 
            A dictionary containing the configuration for the experiment, including:
            - "project" (str): The name of the Weights & Biases (W&B) project.
            - "epochs" (int): Number of training epochs.
            - "batch_size" (int): Size of the mini-batch for training.
            - "lr" (float): Initial learning rate.
            - "optimizer" (str): The optimizer to use ('SGD', 'Adam', etc.).

    Returns:
        None: The function does not return any values, but logs training metrics to W&B and saves the best-performing model locally.
    """
    # Initialize a W&B run with the specified experiment configuration and project name
    with wandb.init(config=exp_config, project=exp_config["project"]) as run:
        
        # Access the W&B configuration for this specific run
        config = wandb.config

        # Print the current W&B configuration
        print("W&B Config:", config)

        # Set the training device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Initialize the YOLO model
        model = YOLO("yolo11m.pt")

        # Add the W&B callback to log training metrics
        add_wandb_callback(model)  # Passa il modello direttamente a add_wandb_callback()

    
        # Start the YOLO model training
        results = model.train(
            data=data_yaml, # Path to the dataset YAML file for training
            epochs=config.epochs,
            device=device,
            imgsz=640,
            project='yolo-eyes',
            batch=config.batch_size,
            lr0=config.lr,
            optimizer=config.optimizer,
            save=True  # Ensure that the trained model is saved
        )


        # Get the path to the best-performing model checkpoint
        best_model_path = f'{results.save_dir}/weights/best.pt'

        # Generate a unique filename using the W&B run ID to save the best model
        unique_model_name = f"/repo/porri/yolo_models/best_model_{run.id}.pt"

        # Copy the best model file to the specified destination
        shutil.copy(best_model_path, unique_model_name)

        print(f"Best YOLO model saved at: {unique_model_name}")


        # Mark the end of the current W&B run
        wandb.finish()


# Entry point
if __name__ == '__main__':

    # Initialize the Roboflow client using the provided API key
    rf = Roboflow(api_key="2PgwSNm3Z6uQgPaynNMR")  # Replace with your actual API key

    # Access the specific Roboflow project and download the dataset in YOLO forma
    try:
        # Takes the dataset and downloads
       # project = rf.project("andreap/eyes3.0-ascvn")
       # dataset = project.version(2).download("yolov8", location="/tmp/roboflow/eye_data/")
        # Set the path to the dataset configuration file (data.yaml)
       # data_yaml = dataset.location + "/data.yaml"
        data_yaml = "/repo/porri/nyst/nyst/roi/yolo_dataset/data.yaml"
    except RuntimeError as e:
        print("Error downloading dataset:", e)
        exit(1)

    # Log into Weights & Biases (W&B) for experiment tracking, ensuring user authentication
    wandb.login()

    # Load the configuration from a YAML file
    with open('/repo/porri/nyst/nyst/roi/config_wb.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Access the parameters section of the sweep configuration
    config = sweep_config["parameters"]
    
    # Set each parameter to its default value (the first listed in 'values') from the configuration
    for param in config:
        config[param] = config[param]["values"][0]

    # Set the project name in the configuration for W&B tracking
    config["project"] = sweep_config["project"]
    
    # Call the training function with the current configuration to start the training process
    train(config)

