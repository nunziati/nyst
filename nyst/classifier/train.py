import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.init as init

from classifier import NystClassifier
from dataset import CustomDataset


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

def initialize_parameters(model, mean=0.0, std=0.02):
    """
    Inizializza i parametri del modello con una distribuzione normale.

    Args:
        model (torch.nn.Module): Il modello i cui parametri devono essere inizializzati.
        mean (float): La media della distribuzione normale.
        std (float): La deviazione standard della distribuzione normale.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            init.normal_(param.data, mean=mean, std=std)
        elif 'bias' in name:
            init.constant_(param.data, 0)


def main():
    # Definisci il tuo modello, criteri di perdita e ottimizzatore
    model = NystClassifier()
    initialize_parameters(model)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Specifica dataset
    data_root = 'data'

    # Specifica augmentation
    train_transform = None
    val_transform = None

    # Crea i dataset
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    train_csv = os.path.join(data_root, 'train_labels.csv')
    val_csv = os.path.join(data_root, 'val_labels.csv')

    train_dataset = CustomDataset(csv_file=train_csv, root_dir=train_dir, transform=train_transform)
    val_dataset = CustomDataset(csv_file=val_csv, root_dir=val_dir, transform=val_transform)

    # Crea i dataloader per il training e la validazione
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Avvia il training
    trained_model, train_stats, val_stats = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda', patience=5)
