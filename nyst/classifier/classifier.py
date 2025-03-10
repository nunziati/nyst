import sys
import os
import torch.nn as nn

# Add 'code' directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nyst.classifier.tunedCNN_time import CNNfeatureExtractorTime

class NystClassifier(nn.Module):
    def __init__(self):
        super(NystClassifier, self).__init__() # Calls up the constructor of the parents class 
        self.input_dim = 150 # Number total frames
        self.num_channels = 8 # Number of positions + speed
        self.nf = 64 # Number of filters
        self.index_activation_middle_layer = 0 # Activation function middle layer
        self.index_activation_last_layer = 0 # Activation function last layer
        
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

    def forward(self,x):# batch_size, 8, 150
        # Pass the input through the tuned CNN to extract features
        features = self.tunedCNN(x)
        # Pass the extracted features through the output network
        output = self.output_net(features)

        return output
    
    
    
    
        
        