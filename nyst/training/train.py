import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold, ParameterGrid
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the desired GPU device (0, 1, 2, ecc.)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Handle memory fragmentation 

# Add 'code' directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.classifier import NystClassifier
from nyst.classifier.dataset import CustomDataset


# Initialisation parameters function
def initialize_parameters(model:nn.Module, mean:float = 0.0, std:float = 0.02): 
    # Initialise model parameters with a normal distribution and bias parameters with zeros const
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param.data, mean=mean, std=std)
        elif 'bias' in name:
            init.constant_(param.data, 0)

### Training Function with k-cross validation and grid-search ### 
def train_model_cross(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1000, patience=30, threshold_correct=0.5): # threshold_correct: probability threshold correct response
 
    # Initialize the best model weights and accuracy
    best_model_wts = model.state_dict() # This dictionary includes all model weights and biases of the best trained model 
    best_acc = 0.0
    
    # Initialize dictionaries to store training and validation statistics
    train_stats = {'loss': [], 'accuracy': []}
    val_stats = {'loss': [], 'accuracy': []}

    # Initialize the early stopping counter
    epochs_no_improve = 0

    # Loop through epochs
    for epoch in range(num_epochs):
                
        # Each epoch has a training phase and a validation phase
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train() # Set the model to training mode
                data_loader = train_loader
            else:
                model.eval() # Set the model to evaluation mode
                data_loader = val_loader

            # Initialize variables to track the total loss and total number of correct predictions during a single training or validation epoch
            running_loss = 0.0
            running_corrects = 0

            # Differentiating between Training and Validation
            if phase == 'Val': # Validation phase
                # Use torch.no_grad() to avoid storing gradients during validation
                with torch.no_grad():
                    for inputs, labels in data_loader:
                        # Move the input and label tensors to the specified device
                        inputs = inputs.to(device)
                        labels = labels.float().to(device)
                        
                        outputs = model(inputs)  # Forward Step
                        loss = criterion(outputs, labels)  # Loss calculation
                        preds = (outputs >= threshold_correct)  # Predictions based on threshold
                        
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
            else: # Training phase
                for inputs, labels in data_loader:
                    inputs = inputs.to(device)
                    labels = labels.float().to(device)
                    
                    optimizer.zero_grad()  # Reset gradients
                    
                    # Forward and backward pass with gradients enabled
                    outputs = model(inputs)  # Forward Step
                    loss = criterion(outputs, labels)  # Loss calculation
                    preds = (outputs >= threshold_correct)  # Predictions
                    
                    loss.backward()  # Backpropagation
                    optimizer.step()  # Optimization step
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

            # Loss and number of correct predictions for each single epoch 
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.float() / len(data_loader.dataset)

            # Store current informations
            if phase == 'Train':
                # Store training loss and accuracy for the current epoch
                train_stats['loss'].append(epoch_loss)
                train_stats['accuracy'].append(epoch_acc)
            else:
                # Store validation loss and accuracy for the current epoch
                val_stats['loss'].append(epoch_loss)
                val_stats['accuracy'].append(epoch_acc)

                # Update counter for early stopping criterion, save the best model weights and other info
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    # If no improvement in accuracy, increment the counter for early stopping
                    epochs_no_improve += 1

                # Check the early stopping criterion (if no improvement for at least 'patience' number of epochs)
                if epochs_no_improve >= patience:
                    print(f'\n\n\nEarly stopping at {epoch}° epoch: {epochs_no_improve} without any accuracy improvement')
                    # model.load_state_dict(best_model_wts)
                    return best_model_wts, train_stats, val_stats, best_acc, best_loss

        # Clear cache after each epoch to free up memory
        del inputs, labels, outputs, preds
        torch.cuda.empty_cache()

        # Print progress at the end of each epoch
        print(f'\t\tEpoch {epoch + 1}/{num_epochs} ------------------- T Loss: {train_stats["loss"][-1]:.4f}, T Acc: {train_stats["accuracy"][-1]:.4f}, V Loss: {val_stats["loss"][-1]:.4f}, V Acc: {val_stats["accuracy"][-1]:.4f}')
    # Print best model for this training
    print(f'\n\tBEST MODEL ------------------- Best V Acc: {best_acc:.4f}, Best V loss: {best_loss:.6f} Best V Epoch: {best_epoch}\n\n')

    return {k: v.cpu() for k, v in best_model_wts.items()}, train_stats, val_stats, best_acc, best_loss

def cross_validate_model(dataset, param_grid, device, save_path, k_folds=4):
    
    # Initialize KFold with the specified number of folds
    kf = KFold(n_splits=k_folds, shuffle=False) # You can set shuffle to True and delete the seed(random_state=42)
    # Initialize list to store all results
    all_results = []
    # Create a parameter grid iterator
    grid = ParameterGrid(param_grid)
  
    print('\n\n')

    # Iterate over all parameter combinations in the parameter grid
    for idx, params in enumerate(tqdm(grid, desc="Percentage of Progress of the entire Training Process (Training using Parameter Grid): ")):
        
        print("\n\n","="*150,"\n")
        print(f"Training Net with this HyperParameters Matrix: {params}")

        # Dictionary to store results for the current parameter set
        fold_results = {'Best models': [], 'Val Loss list': [], 'Val accuracies list': [],'Avarage val loss': 0.0, 'Avarage val accuracy': 0.0}
        
        # List folds models
        fold_models_list = []

        # Perform k-fold cross-validation
        for fold, (train_index, val_index) in enumerate(kf.split(range(len(dataset.tensors[0]))), 1):
            
            print(f"\n\t---> Fold {fold}/{k_folds}:")
            
            # Create training and validation subsets using the generated indices
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)
            
            # Create data loaders for training and validation subsets
            train_loader = DataLoader(train_subset, batch_size=params.get('batch_size', 4), shuffle=False) # 4 is a default value for batch_size if it is not provided in the param_grid
            val_loader = DataLoader(val_subset, batch_size=params.get('batch_size', 4), shuffle=False)
            
            # Re-initialize the model for each fold
            model = NystClassifier().to(device)
            initialize_parameters(model) 
            
            # Select and initialise the optimizer based on parameters
            optimizer_name = params.get('optimizer', 'Adam')
            if optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=params.get('lr', 1e-3), 
                                      momentum= 0.9, 
                                      weight_decay= 1e-4)
            elif optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=params.get('lr', 1e-3))
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")            
            
            # Select and initialise the loss criterion based on parameters
            criterion_name = params.get('criterion', 'BCELoss')
            if criterion_name == 'BCELoss':
                criterion = nn.BCELoss()
            elif criterion_name == 'MSELoss':
                criterion = nn.MSELoss()
            elif criterion_name == 'HingeEmbeddingLoss':
                criterion = nn.HingeEmbeddingLoss()
            else:
                raise ValueError(f"Unsupported criterion: {criterion_name}")
            
            # Extract number of epochs from parameters grid
            num_epochs=params.get('num_epochs', 100)

            # Extract number of epochs from parameters grid
            patience = params.get('patience', 40)
            
            # Extract number of epochs from parameters grid
            threshold_correct = params.get('threshold_correct', 0.5)
            
            # Train the model and retrieve the training and validation statistics
            best_model_wts, _, _, best_acc, best_loss = train_model_cross(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience, threshold_correct)
            
            # Store best models and corrispondent validation accuracies
            fold_models_list.append(best_model_wts)
            fold_results['Val Loss list'].append(best_loss)
            fold_results['Val accuracies list'].append(best_acc)
            
            # Move best model weights to CPU to free GPU memory
            del model  # Free model from GPU memory
            torch.cuda.empty_cache()  # Clear CUDA cache

        # Store the avarage validation accuracy an loss in the partial dictionary
        avg_val_loss = sum(fold_results['Val Loss list']) / k_folds
        avg_val_acc = sum(fold_results['Val accuracies list']) / k_folds
        fold_results['Avarage val loss'] = avg_val_loss
        fold_results['Avarage val accuracy'] = avg_val_acc
        fold_results['Best models'] = fold_models_list[fold_results['Val accuracies list'].index(max(fold_results['Val accuracies list']))]
         
        # Save results for current parameters
        all_results.append({
            'Parameters': params,
            'Param index': idx,
            'Models info': fold_results
        })
        
        print(f"\n\n\nAverage validation accuracy - loss for {idx}° parameters set: {avg_val_acc:.4f} - {avg_val_loss:.4f}")

        # **Empty the GPU cache after each parameter set (after all folds for one param combination)**
        torch.cuda.empty_cache()
    
    # Sort results by average accuracy
    sorted_results = sorted(all_results, key=lambda x: x['Models info']['Avarage val accuracy'], reverse=True)
        
    print("\n\n","="*100,"\n\n")
    print(f"\n\nBest parameters: {sorted_results[0]['Parameters']}, Average validation accuracy: {sorted_results[0]['Models info']['Avarage val accuracy']:.4f}")

    # Extract the list of best models and corresponding validation accuracies
    best_model_param = sorted_results[0]['Models info']['Best models']

    # Create the best model
    best_model = NystClassifier().to(device)
    best_model.load_state_dict(best_model_param)

    # Save the best model
    torch.save(best_model.state_dict(), save_path)

    # Save the model information
    save_model_info(sorted_results, save_path)

    return sorted_results

def save_model_info(results, save_path):
    """
    Save the information of the best model to a text file.
    
    Args:
        results: The results dictionary containing the best model's statistics.
        save_path: The path where to save the text file.
    """
    # Create the file path
    file_path = os.path.join(save_path, 'best_model_info.txt')
    
    # Retrieve best result information
    best_params = results[0]['Parameters']
    best_avg_val_acc = results[0]['Models info']['Average val accuracy']
    best_avg_val_loss = results[0]['Models info']['Average val loss']
    best_model_stats = results[0]['Models info']
    
    # Write the information to the file
    with open(file_path, 'w') as f:
        f.write("Best Model Information\n")
        f.write("="*50 + "\n")
        f.write(f"Best Hyperparameters: {best_params}\n")
        f.write(f"Average Validation Accuracy: {best_avg_val_acc:.4f}\n")
        f.write(f"Average Validation Loss: {best_avg_val_loss:.4f}\n\n")
        
        f.write("Validation Accuracies per Fold:\n")
        for i, acc in enumerate(best_model_stats['Val accuracies list']):
            f.write(f"Fold {i+1}: {acc:.4f}\n")
        
        f.write("\nBest Model Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Best Model Validation Accuracy: {max(best_model_stats['Val accuracies list']):.4f}\n")
        f.write(f"Best Model Validation Loss: {min(best_model_stats['Val Loss list']):.4f}\n")
    
    print(f"\n\nBest model information saved in {file_path}")

# Function to perform the training of the full net 
def training_net(csv_input_file, csv_label_file, save_path, batch_size, lr, optimizer, criterion, threshold_correct, patience, num_epochs, k_folds):
    
    # Define the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create the training and validation datasets object
    dataset = CustomDataset(csv_input_file, csv_label_file)

    # Parameters to be tested during the training phase
    param_grid = {
            'batch_size': batch_size,
            'lr': lr,
            'optimizer': optimizer,
            'criterion': criterion,
            'threshold_correct': threshold_correct,
            'patience': patience,
            'num_epochs': num_epochs,
        }
    

    # Create the training and validation datasets and convert numpy arrays to PyTorch tensors
    train_input_tensor = torch.tensor(dataset.train_signals, dtype=torch.float32)
    train_labels_tensor = torch.tensor(dataset.train_labels, dtype=torch.float32)
               
    # Calculate the number of samples per fold
    n_samples = len(train_input_tensor)
    fold_size = n_samples // k_folds
    
    # Truncate the data to be a multiple of the fold size (avoid different fold size problems)
    train_input_truncated = train_input_tensor[:fold_size * k_folds]
    train_labels_truncated = train_labels_tensor[:fold_size * k_folds]

    # Create Truncate Dataset TensorDataset for train and test sets (avoid different fold size problems)
    train_dataset_truncated = TensorDataset(train_input_truncated, train_labels_truncated)

    # Start the training
    results = cross_validate_model(train_dataset_truncated, param_grid, device, save_path, k_folds)

    return results
