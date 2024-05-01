import torch.nn as nn
from torchviz import make_dot
from torchsummary import summary
import torch.nn.init as init

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder kısmı
        self.encoder = nn.Sequential(
            #input channel arrange
            self.conv_block(3, 64, 1, padding=0),
            
            #1
            nn.MaxPool2d(2, padding=(1, 1)),
            self.conv_block(64, 64, 3),
            
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
            
            #6
            nn.MaxPool2d(2, padding=(1, 0)),
            self.conv_block(64, 64, 3),
            
            #7
            nn.MaxPool2d(2, padding=(0, 0)),
            self.conv_block(64, 64, 3),
            
            #Flatten and FCN
            nn.Flatten(),
            nn.Linear(64 * 2 * 4, 16)
        )

        # Decoder kısmı
        self.decoder = nn.Sequential(
            nn.Linear(16, 64 * 2 * 4),
            nn.Unflatten(1, (64, 2, 4)),
            
            #1
            nn.Upsample(size=(4,8), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #2
            nn.Upsample(size=(6,16), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #3
            nn.Upsample(size=(10,32), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #4
            nn.Upsample(size=(20,64), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #5
            nn.Upsample(size=(38,126), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #6
            nn.Upsample(size=(76,250), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #7
            nn.Upsample(size=(150,498), mode='bicubic', align_corners=True),
            self.conv_block(64, 64, 3),
            
            #output
            nn.Conv2d(64, 3, 1, stride=1)
        )

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed, latent

    def encode(self, x):
        z= self.encoder(x)
        return z

    def decode(self, x):      
        output = self.decoder(x)
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

