import sys
import os
import argparse
import yaml

# Aggiungi la directory 'code' al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.train import training_net
from demo.yaml_function import load_hyperparams, pathConfiguratorYaml


if __name__ == '__main__':
    try:
        ### YAML ###
        _, _, _, _, _, _, _, csv_input_file, csv_label_file, save_path, save_path_info, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds = load_hyperparams(pathConfiguratorYaml) 
        
        # Perform the training and validation of the full net using k-cross validation and grid search
        results = training_net(csv_input_file, csv_label_file, save_path, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds)

    except Exception as e:
        print(f"An error occurred during the Training and Validation phase: {e}")
        exit()