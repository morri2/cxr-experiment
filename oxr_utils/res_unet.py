import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)
    
class EncoderStep(nn.Module):
    """Encodes, Halves width and height"""
    def __init__(self, in_channels, out_channels):
        super(EncoderStep, self).__init__()

        self.doubleconv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        """returns the (<Pooled Output>, <Unpooled/Skip Output>)"""
        skip_con = self.doubleconv(x)
        y = self.pool(skip_con)
        return y, skip_con

class DecoderStep(nn.Module):
    """Decodes (output is 2x width and height of input, but equal to skip_con)"""
    def __init__(self, in_channels, out_channels):
        super(DecoderStep, self).__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.doubleconv = DoubleConv(out_channels * 2, out_channels)
    
    def forward(self, x, skip_con):
        """takes both x and skip"""
        x = self.upconv(x)
        x = torch.cat((skip_con, x), dim=1)
        # TODO: handle non-clean concats (either with output_padding or interpolation) - or not...
        y = self.doubleconv(x)
        return y


class ResUNet(nn.Module):
    
    """Custom dimensioned UNet with "first layer out_channels"=k and "number of steps (enc&dec)=s"""
    def __init__(self, s=4, k=64):
        super(ResUNet, self).__init__()

        self.s = s
        self.k = k
        
        self.encoder_steps = nn.ModuleList(
            [EncoderStep(
                1 if i == 0 else ( k * 2 ** (i-1) ), 
                k * 2 ** i
                ) for i in range(s) ],
        )

        self.bottleneck = DoubleConv(
            k * 2 ** (s-1), 
            k * 2 ** s
            )
        
        self.decoder_steps = nn.ModuleList(
            [DecoderStep(
                k * 2 ** i, 
                k * 2 ** (i-1)
                ) for i in range(s, 0, -1) ],
        )
        
        self.final_conv = nn.Conv2d(k, 1, kernel_size=1)
    

    def name(self):
        return "res-unet-s{}k{}".format(self.s,self.k)

    def forward(self, x):
        skip_cons = []

        # encoder
        for encoder_step in self.encoder_steps:
            x, skip_con = encoder_step(x)
            skip_cons.append(skip_con)
        
        x = self.bottleneck(x)

        # decoder
        skip_cons = skip_cons[::-1]
        for i, decoder_step in enumerate( self.decoder_steps ):
            x = decoder_step(x, skip_cons[i])

        x = self.final_conv(x)
        return x

# Usage
# res = res_unet(input)
# out = clamp(input + res, 0, 1)
# loss = loss_fn(out, lbl)



# Example usage
if __name__ == "__main__":
    import torchinfo


    model = ResUNet(s=4, k=64).to("cuda")

    x = torch.randn((1, 1, 1024, 1024)).to("cuda")  # Example input tensor (Batch, Width, Height)
    
    torchinfo.summary(model, x.shape)

    t = time.time()
    out = model(x)
    print("time =", time.time() - t)
    print("in shape", out.shape)  # Should be (1, 256, 256)
    print("out shape", x.shape)
    #import matplotlib.pyplot as plt
    #plt.imshow(out.detach().numpy(). squeeze() , cmap="gray")
    #plt.show()