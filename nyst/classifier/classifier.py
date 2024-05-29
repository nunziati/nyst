from tunedCNN_time import CNNfeatureExtractorTime
import torch.nn as nn


class NystClassifier(nn.Module):
    def __init__(self):
        super(CNNfeatureExtractorTime, self).__init__()
        self.input_dim=320
        self.num_channels=8
        self.nf=64
        self.index_activation_middle_layer=0 
        self.index_activation_last_layer=0
        self.tunedCNN = CNNfeatureExtractorTime(self.input_dim, self.num_channels, self.nf, self.index_activation_middle_layer, self.index_activation_last_layer)
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

    def forward(self,x):# batch_size, 8, 320
        
        features = self.tunedCNN(x)
        output = self.output_net(features)

        return output
    
    
    
    
        
        