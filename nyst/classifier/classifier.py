import sys
import os
import torch
import torch.nn as nn

# Add 'code' directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.tunedCNN_time import CNNfeatureExtractorTime

class NystClassifier(nn.Module):
    def __init__(self, input_dim=150, num_channels=8, nf=8, index_activation_middle_layer=0, index_activation_last_layer=0):
        super(NystClassifier, self).__init__() # Calls up the constructor of the parents class 
        self.input_dim = input_dim # Number total frames
        self.num_channels = num_channels # Number of positions + speed
        self.nf = nf # Number of filters
        self.index_activation_middle_layer = index_activation_middle_layer # Activation function middle layer
        self.index_activation_last_layer = index_activation_last_layer # Activation function last layer
        
        # Initialize the tuned CNN and the output network
        self.tunedCNN = CNNfeatureExtractorTime(self.input_dim, self.num_channels, self.nf, self.index_activation_middle_layer, self.index_activation_last_layer)
        
        # Initialize the output fully connected network
        self.output_net = nn.Sequential(

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def from_pretrained(path):
        # Load the provided weights from a file
        checkpoint = torch.load(path)
        weights = checkpoint['weights']
        params = checkpoint['params']

        # Create an instance of NystClassifier using the provided parameters
        classifier = NystClassifier(
            input_dim=params['input_dim'],
            num_channels=params['num_channels'],
            nf=params['nf'],
            index_activation_middle_layer=params['index_activation_middle_layer'],
            index_activation_last_layer=params['index_activation_last_layer']
        )
        # Load the provided weights into the classifier
        classifier.load_state_dict(weights)
        classifier.eval()
        
        return classifier

    def forward(self, x):# batch_size, 8, 150
        # Pass the input through the tuned CNN to extract features
        features = self.tunedCNN(x)
        # Pass the extracted features through the output network
        output = self.output_net(features)

        return output
    
    def load_weights(self, path):
        # Load the weights for the tuned CNN and the output network
        self.load_state_dict(torch.load(path))
        self.eval()
    
    def save_weights(self, path):
        # Save the weights for the tuned CNN and the output network
        weights = self.state_dict()
        params = {
            'input_dim': self.input_dim,
            'num_channels': self.num_channels,
            'nf': self.nf,
            'index_activation_middle_layer': self.index_activation_middle_layer,
            'index_activation_last_layer': self.index_activation_last_layer
        }

        torch.save({'weights': weights, 'params': params}, path)
        print(f"Model saved to {path}")
    
    
    
        
        