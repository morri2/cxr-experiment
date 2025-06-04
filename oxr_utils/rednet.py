import math
from torch import nn


class REDNet(nn.Module):
    def __init__(self, num_layers=15, num_features=64, num_channels=1, skip_stride=2):
     
        super(REDNet, self).__init__()
        self.num_layers = num_layers
        self.num_features = num_features
        self.num_channels = num_channels
        self.skip_stride = skip_stride

        self.encoder_layers = nn.ModuleList([])

        self.encoder_layers.append(
            nn.Sequential(
                    nn.Conv2d(num_channels, num_features, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True)))
        
        for i in range(num_layers - 1):
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)))
        
            
        self.decoder_layers = nn.ModuleList([])
        for i in range(num_layers - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(inplace=True)
                ))
        self.decoder_layers.append(
            nn.Sequential(
                    nn.ConvTranspose2d(num_features, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True)))

        self.final_activation = nn.ReLU(inplace=True)
        
    @property
    def name(self):
        return f"RED{self.num_layers * 2}" 

    def forward(self, x):
        input = x

        skips = []

        # Encoder
        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)
            if (i + 1) % self.skip_stride == 0 and i < self.num_layers - 1:
                skips.append(x)

        skips.reverse()
    
        # Decoder
        skip_idx = 0
        for i in range(self.num_layers):
            x = self.decoder_layers[i](x)
            if (i + 1) % self.skip_stride == 0 and skip_idx < len(skips):
                x = x + skips[skip_idx]
                skip_idx += 1
                
        x = input + x # input + residual
        x = self.final_activation(x)
        return x




if __name__ == "__main__":
    import time
    import torch  
    import torchinfo

    model = REDNet(num_layers=10, num_features=64, num_channels=1, skip_stride=2).to("cuda")
    torchinfo.summary(model, input_size=(1, 1, 1024, 1024), device="cuda")


    # for num_layers in [10, 20, 30]:
    #     model = REDNet(num_layers=num_layers, num_features=64, num_channels=1, skip_stride=2).to("cuda")
    #     model(torch.randn(1, 1, 1024, 1024).to("cuda")) # warm up

    #     with torch.no_grad():
    #         input_tensor = torch.randn(1, 1, 1024, 1024).to("cuda")
    #         t = time.time()
    #         for i in range(10):
    #             print(i)
    #             output_tensor = model(input_tensor)
    #     print(f"{model.name} (cuda): {(time.time() - t) / 10:.4f} seconds, output shape: {output_tensor.shape}")


    # for num_layers in [10, 20, 30]:
    #     model = REDNet(num_layers=num_layers, num_features=64, num_channels=1, skip_stride=2)
    #     model = model.to("cpu")

    #     with torch.no_grad():
    #         input_tensor_cpu = torch.randn(1, 1, 1024, 1024).to("cpu")  # Example input tensor
    #         t = time.time()
    #         output_tensor = model(input_tensor_cpu)
    #     print(f"RED{num_layers} (cpu): {time.time() - t:.4f} seconds, output shape: {output_tensor.shape}")
