
import torch
import torch.nn as nn
import torch.nn.functional as F

class KSpaceUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, num_features=64):
        super(KSpaceUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t):
        t_embed = F.one_hot(t, num_classes=self.timesteps).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x + t_embed
        x = self.encoder(x)
        x = self.decoder(x)
        return x