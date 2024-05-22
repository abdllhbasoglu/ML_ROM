# This model consists of 5 Layer, 'Bicubic' interpolation for upsampling and Latent Space = 16. Rest is similar to baseline model 

import torch.nn as nn
import torch.nn.init as init

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder kısmı
        self.encoder = nn.Sequential(
            #input channel arrange
            self.conv_block(3, 32, 1, padding=0),
            
            #1
            nn.MaxPool2d(2, padding=(1, 1)),
            self.conv_block(32, 64, 3),
            
            #2
            nn.MaxPool2d(2, padding=(0, 1)),
            self.conv_block(64, 64, 3),
            
            #3
            nn.MaxPool2d(2, padding=(1, 1)),
            self.conv_block(64, 64, 3),
            
            #4
            nn.MaxPool2d(2, padding=(0, 0)),
            self.conv_block(64, 64, 3),
            
            #5
            nn.MaxPool2d(2, padding=(1, 0)),
            self.conv_block(64, 64, 3),
            
            
            #Flatten and FCN
            nn.Flatten(),
            
        )

        self.l1 = nn.Linear(64 * 6 * 16, 16)  # first linear layer for encoder part
        self.l2 = nn.Linear(16, 64 * 6 * 16) # second linear layer for decoder part
        
        # Decoder kısmı
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 6, 16)),
                                    
            #1
            nn.Upsample(size=(10,32), mode='bicubic'),
            self.convTranspose_block(64, 64, 3),
                        
            #2
            nn.Upsample(size=(20,64), mode='bicubic'),
            self.convTranspose_block(64, 64, 3),
                 
            #3
            nn.Upsample(size=(38,126), mode='bicubic'),
            self.convTranspose_block(64, 64, 3),
                        
            #4
            nn.Upsample(size=(76,250), mode='bicubic'),
            self.convTranspose_block(64, 64, 3),
                                   
            #5
            nn.Upsample(size=(150,498), mode='bicubic'),
            self.convTranspose_block(64, 32, 3),
                                   
            #output
            nn.Conv2d(32, 3, 1, stride=1)
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def encode(self, x):
        h= self.encoder(x)
        output = self.l1(h)
        return output

    def decode(self, x):
        z = self.l2(x)        
        output = self.decoder(z)
        return output

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
    def convTranspose_block(self, in_channels, out_channels, kernel_size, stride=1, padding=1, Out_P=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, Out_P),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    #Weight initialize etmek icin
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
                    

