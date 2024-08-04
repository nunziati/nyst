import os
import copy
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.init as init
from sklearn.model_selection import KFold, ParameterGrid
from classifier import NystClassifier
from dataset import CustomDataset


### Classical Training Function ###
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', patience=5):
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_stats = {'loss': [], 'accuracy': []}
    val_stats = {'loss': [], 'accuracy': []}

    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Ogni epoca ha una fase di training e una di validazione
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Imposta il modello in modalità training
                data_loader = train_loader
            else:
                model.eval()  # Imposta il modello in modalità evaluation
                data_loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Itera sui dati
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset dei gradienti
                optimizer.zero_grad()

                # Solo nella fase di training esegue la forward e backward pass
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = outputs >= 0.5

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # Statistiche
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'Train':
                train_stats['loss'].append(epoch_loss)
                train_stats['accuracy'].append(epoch_acc.item())
            else:
                val_stats['loss'].append(epoch_loss)
                val_stats['accuracy'].append(epoch_acc.item())

                # Controlla per l'early stopping
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    print('Early stopping')
                    model.load_state_dict(best_model_wts)
                    return model, train_stats, val_stats

    print(f'Best val Acc: {best_acc:.4f}')

    # Carica i pesi migliori del modello
    model.load_state_dict(best_model_wts)


    return model, train_stats, val_stats


### Training Function with k-cross validation and grid-search ### 
def train_model_cross(model, train_loader, val_loader, criterion, optimizer, num_epochs=1000  , device='cuda', patience=30, threshold_correct=0.5): # threshold_correct: probability threshold correct response
   
    # Move the model to the specified device (e.g., GPU or CPU)
    model = model.to(device)

    # Initialize the best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict()) # This dictionary includes all model weights and biases of the best trained model 
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

            # Initialise variables to track the total loss and total number of correct predictions during a single training or validation epoch
            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data
            for inputs, labels in data_loader:
                # Move the input and label tensors to the specified device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Reset gradients
                optimizer.zero_grad()

                # Forward and backward pass only in training phase
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs) # Forward
                    loss = criterion(outputs, labels) # Loss calculation
                    preds = outputs >= threshold_correct # Tensore of correct/false predictions based on a specific threshold
                    
                    # Backprop step and optimization step 
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                # Partial Statistics across the inputs
                running_loss += loss.item() * inputs.size(0) # loss of batch * size of first tensor axis (size of mini-batch) = total loss for current batch
                running_corrects += torch.sum(preds == labels.data) # Accumulate the number of correct predictions of the batches

            # Loss and number of correct predictions for each single epoch 
            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'Epoch {epoch}/{num_epochs - 1}','-' * 10, f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'Train':
                # Store training loss and accuracy for the current epoch
                train_stats['loss'].append(epoch_loss)
                train_stats['accuracy'].append(epoch_acc.item())
            else:
                # Store validation loss and accuracy for the current epoch
                val_stats['loss'].append(epoch_loss)
                val_stats['accuracy'].append(epoch_acc.item())

                # Check for early stopping and count the number of "no improvement in accuracy"
                if epoch_acc > best_acc:
                    # If current epoch accuracy is better than the best accuracy, update best accuracy and save the current model weights
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                else:
                    # If no improvement in accuracy, increment the counter for early stopping
                    epochs_no_improve += 1

                # If no improvement for 'patience' number of epochs, stop training early
                if epochs_no_improve >= patience:
                    print('\n\n\nEarly stopping at {epoch}° epoch: {epochs_no_improve} without any accuracy improvement')
                    #model.load_state_dict(best_model_wts)
                    return best_model_wts, train_stats, val_stats, best_acc, best_loss


        # Print progress at the end of each epoch
        print(f'\t\tEpoch {epoch + 1}/{num_epochs} ------------------- T Loss: {train_stats["loss"][-1]:.4f}, T Acc: {train_stats["accuracy"][-1]:.4f}, V Loss: {val_stats["loss"][-1]:.4f}, V Acc: {val_stats["accuracy"][-1]:.4f}')
    # Print best model for this training
    print(f'\n\tBEST MODEL ------------------- Best val Acc: {best_acc:.4f}, Best loss: {best_loss:.6f} Best Epoch: {best_epoch}\n\n')

    # Load the model's best weights
    #model.load_state_dict(best_model_wts)

    return best_model_wts, train_stats, val_stats, best_acc, best_loss

def cross_validate_model(model_class, dataset, param_grid, device='cuda:0', k_folds=5, save_path='D:/nyst_labelled_videos/best_model.pth'):
    
    # Initialize KFold with the specified number of folds and whether to shuffle the data before splitting.
    kf = KFold(n_splits=k_folds, shuffle=False) # You can set shuffle to True
    results = [] # Store k-cross results

    # Iterate over all parameter combinations in the parameter grid
    for params in tqdm(ParameterGrid(param_grid), desc="Parameter Grid"):
        print("\n\n","-"*100,"\n\n")
        print(f"Training Net with this Parameters Matrix: {params}\n\n")

        # Initialize a dictionary to store results for the current parameter set
        fold_results = {'Parameters set': params, 'Best models': [], 'Val accuracies list': [], 'Avarage val accuracy': 0.0}
        
        # Perform k-fold cross-validation
        for fold, (train_index, val_index) in enumerate(kf.split(dataset), 1):
            print(f"\n\tTraining fold {fold}/{k_folds}:")
            # Create training and validation subsets using the generated indices
            train_subset = Subset(dataset, train_index)
            val_subset = Subset(dataset, val_index)
            
            # Create data loaders for training and validation subsets
            train_loader = DataLoader(train_subset, batch_size=params.get('batch_size', 16), shuffle=False) # 16 is a default value for batch_size if it is not provided in the param_grid
            val_loader = DataLoader(val_subset, batch_size=params.get('batch_size', 16), shuffle=False)
            
            # Initialize the model
            model = model_class
            
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
            best_model_wts, _, _, best_acc, _ = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience, threshold_correct)
            
            # Store best models and corrispondent validation accuracies
            fold_results['Val accuracies list'].append(best_acc)
            fold_results['Best models'].append(best_model_wts)

        # Store the mean validation accuracy
        avg_val_acc = sum(fold_results['Val accuracies list']) / k_folds
        fold_results['Avarage val accuracy'] = avg_val_acc
        
        # Save the result for each set of parameters 
        results.append(fold_results)
        
        print(f"\n\n\nAverage validation accuracy for parameters {params}: {avg_val_acc:.4f}")

    # Sort results by average accuracy in validation
    results = sorted(results, key=lambda x: x['avg_val_accuracy'], reverse=True)
    
    print("\n\n","-"*100,"\n\n")
    print(f"\n\nBest parameters: {results[0]['Parameters set']}, Average validation accuracy: {results[0]['avg_val_accuracy']:.4f}")

    # Extract the list of best models and corresponding validation accuracies
    best_models_list = results[0]['Best models']
    best_val_accuracies_list = results[0]['Val accuracies list']

    # Find the model with the highest validation accuracy
    best_val_acc = max(best_val_accuracies_list)
    best_model_index = best_val_accuracies_list.index(best_val_acc)
    best_model_par = best_models_list[best_model_index]

    # Create the beste model
    best_model = model_class()
    best_model.load_state_dict(best_model_par)

    # Save the best model
    torch.save(best_model.state_dict(), save_path)
    
    return results, best_model

# Initialisation parameters function
def initialize_parameters(model:nn.Module, mean:float = 0.0, std:float = 0.02): 
    # Initialise model parameters with a normal distribution
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param.data, mean=mean, std=std)
        elif 'bias' in name:
            init.constant_(param.data, 0)

def main_classic():
    # Define our model, loss criteria and optimiser
    model = NystClassifier()
    initialize_parameters(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Specify the dataset path
    data_root = 'data'

    # Specifiy augmentation
    train_transform = None
    val_transform = None

    # Create the training and validation datasets paths
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    train_csv = os.path.join(data_root, 'train_labels.csv')
    val_csv = os.path.join(data_root, 'val_labels.csv')

    # Create the training and validation datasets
    train_dataset = CustomDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    val_dataset = CustomDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform)

    # Create the dataloader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Start the training
    trained_model, train_stats, val_stats = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', patience=5)

# Main for the cross-validation version 
def main():
    # Define our model, loss criteria and optimiser
    model = NystClassifier()
    initialize_parameters(model)
    
    # Specify the dataset path
    data_root = 'D:/nyst_labelled_videos'

    # Specifiy augmentation
    train_transform = None
    val_transform = None
    dataset = CustomDataset(csv_file=os.path.join(data_root, 'labels.csv'), root_dir=os.path.join(data_root, 'videos'), transform=transform)

    param_grid = {
            'batch_size': [16, 32, 64],
            'lr': [0.001, 0.0001],
            'optimizer': ['Adam', 'SGD'],
            'criterion': ['BCELoss','MSELoss'],
            'threshold_correct': [0.5, 0.6, 0.7],
            'patience': [5, 10],
            'num_epochs': [50, 100],
        }

    # Create the training and validation datasets paths
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    train_csv = os.path.join(data_root, 'train_labels.csv')
    val_csv = os.path.join(data_root, 'val_labels.csv')

    # Create the training and validation datasets
    train_dataset = CustomDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    val_dataset = CustomDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform)

    # Create the dataloader for training and validation datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Start the training
    results, best_params = cross_validate_model(model_class, dataset, param_grid, device='cuda', k_folds=5)

# Start the code execution
if __name__ == "__main__":
    main()