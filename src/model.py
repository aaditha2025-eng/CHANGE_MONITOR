import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SiameseUNet, self).__init__()
        
        # Shared Encoder (Siamese branch)
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Decoder (Takes concatenated features from both branches)
        # 512 * 2 = 1024 input channels at the bottleneck
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        
        self.dec3 = DoubleConv(256 + 256*2, 256) # up3 + enc3_before + enc3_after
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        
        self.dec2 = DoubleConv(128 + 128*2, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        
        self.dec1 = DoubleConv(64 + 64*2, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)

    def forward_once(self, x):
        # Encoder forward pass
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        return e1, e2, e3, e4

    def forward(self, before, after):
        # Pass both images through the shared encoder
        b1, b2, b3, b4 = self.forward_once(before)
        a1, a2, a3, a4 = self.forward_once(after)
        
        # Bottleneck: Concatenate the deepest features
        bottleneck = torch.cat([b4, a4], dim=1) # 512 + 512 = 1024
        
        # Decoder pass with skip connections from BOTH branches
        d4 = self.dec4(bottleneck)
        
        d3 = self.up3(d4)
        # Handle potential size mismatch due to pooling/unpooling
        if d3.size() != b3.size():
            d3 = F.interpolate(d3, size=b3.shape[2:])
        d3 = torch.cat([d3, b3, a3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.size() != b2.size():
            d2 = F.interpolate(d2, size=b2.shape[2:])
        d2 = torch.cat([d2, b2, a2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.size() != b1.size():
            d1 = F.interpolate(d1, size=b1.shape[2:])
        d1 = torch.cat([d1, b1, a1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return torch.sigmoid(out) # Return probabilities [0, 1]
