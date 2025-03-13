import copy
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold
import os
from demo.yaml_function import load_hyperparams, pathConfiguratorYaml
from nyst.classifier.classifier import NystClassifier
from nyst.dataset.dataset import NystDataset

# Set the desired GPU device and manage CUDA memory fragmentation
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set desired GPU device
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Handle memory fragmentation

# Initialisation parameters function
def initialize_parameters(model:nn.Module, mean:float = 0.0, std:float = 0.02): 
    """
    Initializes model parameters with a normal distribution and bias parameters with zeros.
    
    Args:
        model (nn.Module): The model to initialize.
        mean (float): The mean of the normal distribution for weights. Default is 0.0.
        std (float): The standard deviation of the normal distribution for weights. Default is 0.02.
    
    Returns:
        None: Modifies the model in place.
    """
    # Loop through all named parameters in the model
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param.data, mean=mean, std=std)
        elif 'bias' in name:
            init.constant_(param.data, 0)

# Function to get the optimizer and loss criterion based on the configuration
def get_optimizer_and_criterion(model, config):
    """
    Creates an optimizer and loss function based on the provided configuration.
    
    Args:
        model (nn.Module): The model for which to create the optimizer.
        config: Configuration object containing optimizer and criterion settings.
    
    Returns:
        optimizer: The initialized optimizer for the model.
        criterion: The initialized loss function for training.
    """
    # Select optimizer based on configuration
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

    # Select loss criterion based on configuration
    if config.criterion == 'BCELoss':
        criterion = nn.BCELoss()
    elif config.criterion == 'MSELoss':
        criterion = nn.MSELoss()

    return optimizer, criterion

# Training function with k-fold cross-validation
def train_model_cross(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1000, patience=30, threshold_correct=0.5):
    """
    Trains the model with k-fold cross-validation.
    
    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        criterion: Loss function for training.
        optimizer: Optimizer for updating model weights.
        device: Device to run the model on (CPU or GPU).
        num_epochs (int): Number of epochs for training. Default is 1000.
        patience (int): Number of epochs to wait for improvement before stopping. Default is 30.
        threshold_correct (float): Threshold for predicting correct labels. Default is 0.5.
    
    Returns:
        model: The trained model with the best weights.
    """
    best_model_wts = model.state_dict() # Save the best model weights
    best_acc = 0.0 # Initialize best accuracy
    epochs_no_improve = 0 # Initialize counter for epochs without improvement

    # Loop through each epoch
    for epoch in range(num_epochs):

        # Set model to training/validation mode
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0 # Initialize running loss
            running_corrects = 0 # Initialize correct predictions count

            # Loop through data batches
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.float().to(device).view(-1, 1)
                optimizer.zero_grad() # Clear previous gradients

                # Enable gradients only during training
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute loss
                    preds = (outputs >= threshold_correct)  # Compute predictions

                    # If training phase execute backward pass and update model weights
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # Accumulate loss and Count correct predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.float() == labels.data)

            # Compute average loss and accuracy for the epoch
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.float() / len(data_loader.dataset)

            # Log metrics to W&B
            wandb.log({
                f"{phase}_loss": epoch_loss,   # Log loss for the current phase
                f"{phase}_accuracy": epoch_acc, # Log accuracy for the current phase
                "epoch": epoch # Log current epoch
            })

            # Check for improvements in validation accuracy
            if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                epochs_no_improve = 0
            elif phase == 'Val':
                epochs_no_improve += 1

            # Early stopping if no improvement
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                wandb.log({"early_stopping_epoch": epoch}) # Log early stopping epoch
                model.load_state_dict(best_model_wts) # Load best model weights
                return model, best_acc
            
        # Clear CUDA cache
        torch.cuda.empty_cache()

    return model, best_acc

# Cross-validation and hyperparameter sweep function
def cross_validate_model(dataset, config, device, save_path_wb, k_folds=5):
    """
    Performs cross-validation and hyperparameter sweep on the model.
    
    Args:
        dataset: The dataset to train and validate on.
        config: Configuration object containing training settings.
        device: Device to run the model on (CPU or GPU).
        save_path (str): Path to save the best model weights.
        k_folds (int): Number of folds for cross-validation. Default is 4.
    
    Returns:
        None: Saves the best model weights.
    """
    # Initializations
    kf = KFold(n_splits=k_folds, shuffle=True) # KFold object
    best_avg_acc = 0.0 # Variable to store best average accuracy
    best_model_wts = None # Variable to store best model weights

    '''# Using only the training data for cross-validation
    train_signals = dataset.train_signals
    train_labels = dataset.train_labels'''

    # Loop through each fold
    for fold, (train_index, val_index) in enumerate(kf.split(range(len(dataset.tensors[0]))), 1):        
        
        # Create training and validation subset
        train_subset = Subset(dataset, train_index)
        val_subset = Subset(dataset, val_index)

        # Create DataLoader for training and validation
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        # Initialize model and model parameters
        model = NystClassifier().to(device)
        initialize_parameters(model)

        # Get optimizer and criterion
        optimizer, criterion = get_optimizer_and_criterion(model, config)

        # Monitor gradients with wandb.watch
        wandb.watch(model, criterion, log="gradients")

        # Train the model for the current fold
        best_model, fold_acc = train_model_cross(model, train_loader, val_loader, criterion, optimizer, device, config.epochs, config.patience, config.threshold_correct)

        # Log GPU/CPU stats to W&B
        wandb.log({"GPU_memory_allocated": torch.cuda.memory_allocated(), "CPU_usage": os.cpu_count()})

        # Update best model if accuracy improves
        if fold_acc > best_avg_acc:
            best_avg_acc = fold_acc # Update best average accuracy
            best_model_wts = copy.deepcopy(best_model.state_dict())  # Copy best model weights
            print(f"New best model found for fold {fold} with accuracy {fold_acc}")

    
    # Save the model with the best accuracy across all folds
    dir_name = wandb.run.dir.split('/')[-2] 
    model_dir = os.path.join(save_path_wb, dir_name)

    os.makedirs(model_dir, exist_ok=True)

    final_model_save_path = os.path.join(model_dir, 'best_model.pth')
    torch.save(best_model_wts, final_model_save_path)
    print(f"Saved best model with accuracy {best_avg_acc} at {final_model_save_path}")

    wandb.finish()

        
        

# Main function to start training with W&B sweep
def train(config=None):
    """
    Starts the training process with W&B integration.
    Initializes the W&B run, sets up the device (CPU or GPU), loads the dataset, and starts
    the cross-validation process using the provided configuration.

    Args:
        config: W&B sweep configuration object (optional). Contains parameters like learning rate,
                optimizer type, and number of epochs. If None, W&B will provide the sweep config.
    Returns:
        None
    """
    # Load
    _, _, _, _, _, _, _, csv_file_train, _, _, _, _, _, _, save_path_wb = load_hyperparams(pathConfiguratorYaml) 

    # Initialize W&B with the given configuration
    with wandb.init(config=config):
        config = wandb.config  # Access the W&B configuration settings
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        dataset = NystDataset(csv_file_train)  # Path to your dataset

        # Create the training and validation datasets and convert numpy arrays to PyTorch tensors
        train_input_tensor = torch.tensor(dataset.fil_norm_data, dtype=torch.float32)
        train_labels_tensor = torch.tensor(dataset.extr_data['labels'], dtype=torch.float32)
                
        # Calculate the number of samples per fold
        n_samples = len(train_input_tensor)
        fold_size = n_samples // 5
        
        # Truncate the data to be a multiple of the fold size (avoid different fold size problems)
        train_input_truncated = train_input_tensor[:fold_size * 5]
        train_labels_truncated = train_labels_tensor[:fold_size * 5]

        # Create Truncate Dataset TensorDataset for train and test sets (avoid different fold size problems)
        train_dataset_truncated = TensorDataset(train_input_truncated, train_labels_truncated)

        # Start cross-validation
        cross_validate_model(train_dataset_truncated, config, device, save_path_wb, k_folds=5)
