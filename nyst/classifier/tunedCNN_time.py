import torch.nn as nn
import torch

# Activation functions list:
Activation_list = [nn.ReLU(inplace=True), nn.Tanh()]

class CNNfeatureExtractorTime(nn.Module):
    def __init__(self, input_dim, num_channels, nf, index_activation_middle_layer=0, index_activation_last_layer=-1):
        super(CNNfeatureExtractorTime, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels  # Number of desired channels
        self.nf = nf  # Number of filters
        self.index_activation_middle_layer = index_activation_middle_layer  # Index for selecting the activation function of middle layers
        self.index_activation_last_layer = index_activation_last_layer  # Index for selecting the activation function of last layer

        # Defining the structure of the network
        self.main = nn.Sequential(
            # The first layer - convolutional
            nn.Conv1d(in_channels=self.num_channels, out_channels=self.nf, kernel_size=3, stride=1, padding=1, bias=True),
            Activation_list[self.index_activation_middle_layer],
            nn.MaxPool1d(kernel_size=2, stride=2),

            # The second layer - convolutional
            nn.Conv1d(in_channels=self.nf, out_channels=self.nf*2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.nf*2),
            Activation_list[self.index_activation_middle_layer],
            nn.MaxPool1d(kernel_size=2, stride=2),

            # The third layer - convolutional
            nn.Conv1d(in_channels=self.nf*2, out_channels=self.nf*4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.nf*4),
            Activation_list[self.index_activation_middle_layer],

            # The fourth layer - convolutional
            nn.Conv1d(in_channels=self.nf*4, out_channels=self.nf*8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.nf*8),
            Activation_list[self.index_activation_middle_layer],

            # The fifth layer - convolutional
            nn.Conv1d(in_channels=self.nf*8, out_channels=self.nf*16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(self.nf*16),
            Activation_list[self.index_activation_last_layer],
            nn.Flatten()
        )

        # Calculate the number of features after the convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.num_channels, self.input_dim)
            dummy_output = self.main(dummy_input)
            num_features = dummy_output.shape[1]

        # Last layer - fully connected
        self.fc = nn.Sequential(
            nn.Linear(in_features=num_features, out_features=2048),
            nn.Linear(in_features=2048, out_features=256)
        )

    def forward(self, input):
        conv_output = self.main(input)
        return self.fc(conv_output)