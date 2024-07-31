import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
       # image = read_image(img_name)  # Definisci la tua funzione read_image
        label = self.data_frame.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_custom_dataset(csv_file, root_dir, batch_size=32, transform=None):
    """
    Carica un dataset personalizzato da un file CSV e una directory.

    Args:
        csv_file (str): Il percorso al file CSV con le annotazioni.
        root_dir (str): Il percorso alla directory con tutte le immagini.
        batch_size (int): La dimensione del batch per il dataloader.
        transform (torchvision.transforms.Compose): Le trasformazioni da applicare alle immagini.

    Returns:
        DataLoader: Un dataloader per il dataset personalizzato.
    """
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    
    return dataset

# Esempio di utilizzo:
train_loader = load_custom_dataset('path/to/train_labels.csv', 'path/to/train', batch_size=32)
val_loader = load_custom_dataset('path/to/val_labels.csv', 'path/to/val', batch_size=32)
