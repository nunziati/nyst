import sys
import os
from yaml_function import *
import wandb

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.train_wb import train
from demo.yaml_function import load_hyperparams, pathConfiguratorYaml

if __name__ == "__main__":

    # Logs into the Weights and Biases (W&B) platform, ensuring the user is authenticated
    wandb.login()

    # Load the sweep configuration from a YAML file
    with open('sweep_config.yaml') as file:
        sweep_config = yaml.safe_load(file)

    # Create a sweep in W&B using the configuration loaded
    sweep_id = wandb.sweep(sweep_config, project="nyst_detection")

    # Launches the W&B agent to run the sweep
    wandb.agent(sweep_id, train, count=100)