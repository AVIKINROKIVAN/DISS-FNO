import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Инициализация с меньшим масштабом
        scale = (0.1 / (in_channels * out_channels))**0.5
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        
        # Удален Dropout для комплексных весов

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x.float())  # Явное приведение к float32
        
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        
        # Без Dropout для комплексных весов
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, :self.modes1, :self.modes2], 
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", 
            x_ft[:, :, -self.modes1:, :self.modes2], 
            self.weights2
        )
        
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        
        self.input_layer = nn.Sequential(
            nn.Linear(1, width),
            nn.GELU(),
            nn.Dropout(0.1),  # Dropout остается для вещественных чисел
            nn.LayerNorm(width),
            nn.Linear(width, width)
        )
        
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(3)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(3)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm2d(width) for _ in range(3)
        ])
        
        # Добавляем дополнительный Dropout после каждого Fourier+Conv блока
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(3)
        ])
        
        self.output_layer = nn.Sequential(
            nn.Linear(width, width*2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width*2, 1)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.input_layer(x)
        x = x.permute(0, 3, 1, 2)
        
        for fourier, conv, norm, dropout in zip(self.fourier_layers, self.conv_layers, self.norms, self.dropouts):
            x_fourier = fourier(x)
            x_conv = conv(x)
            x = F.gelu(norm(x_fourier + x_conv))
            x = dropout(x)  # Применяем Dropout после каждого блока
        
        x = x.permute(0, 2, 3, 1)
        return self.output_layer(x)