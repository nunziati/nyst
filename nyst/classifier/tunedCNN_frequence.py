import torch.nn as nn

#Activation functions list:
Activation_list = [nn.ReLU(inplace=True), nn.Tanh()]

class CNNfeatureExtractorTime(nn.Module):
    def __init__(self, input_dim, num_channels, nf, index_activation_middle_layer=0, index_activation_last_layer=-1):
        super(CNNfeatureExtractorTime, self).__init__()
        self.input_dim = input_dim
        self.num_channels = num_channels # Number of desire channels 
        self.nf = nf # Numbe filters
        self.index_activation_middle_layer = index_activation_middle_layer # Index for selecting the activation function of middle layers
        self.index_activation_last_layer = index_activation_last_layer # Index for selecting the activation function of last layer
        

        
        #Defining the structure of the network. So "nn.Sequential" is used to define the architecture of a sequential network by 
        #specifying the order and type of layers that make it up.
    
        self.main = nn.Sequential(
            
            # The first layer - convolutional
            nn.Conv1d(in_channels=self.num_channels, out_channels=self.nf, kernel_size=3, stride=1, bias=True), #convolution
            Activation_list[self.index_activation_middle_layer], # apply the activation function to the batch normalization results
            
            # The second layer - Pooling
            nn.MaxPool1d(kernel_size=2, stride=2), # Dopo il layer di attivazione, aggiungi il max pooling per ridurre la dimensionalità
            
            
            # The third layer - convolutional
            nn.Conv1d(in_channels=self.ngf, out_channels=self.nf*2, kernel_size=3, stride=1, bias=True), #convolution
            nn.BatchNorm1d(self.nf*2),
            Activation_list[self.index_activation_middle_layer],

            # The second layer - Pooling
            nn.MaxPool1d(kernel_size=2, stride=2),
            
            # The fourth layer - convolutional
            nn.Conv1d(in_channels=self.nf*2, out_channels=self.nf*4, kernel_size=3, stride=1, bias=True), #convolution
            nn.BatchNorm1d(self.nf*4),
            Activation_list[self.index_activation_middle_layer],

            # The fifth layer - convolutional
            nn.Conv1d(in_channels=self.nf*4, out_channels=self.nf*8, kernel_size=3, stride=1, bias=True), 
            nn.BatchNorm1d(self.nf*8),
            Activation_list[self.index_activation_middle_layer],


            #The sixth layer - convolutional
            nn.Conv1d(in_channels=self.ngf*8, out_channels=self.nf*16, kernel_size=3, stride=1, bias=True),
            nn.BatchNorm1d(self.nf*16),
            Activation_list[self.index_activation_last_layer],
            nn.Flatten(),  # Flatten the output before passing to fully connected layers
            
            #Last layer - fully connected
            nn.Linear(in_features=self.nf*16*self.input_dim/4, out_features=2048),
            nn.Linear(in_features=2048, out_features=256)

        )

    def forward(self, input):
        return self.main(input)