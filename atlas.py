from torch.nn import Module, Conv2d, LeakyReLU, Sequential, Dropout2d, Sigmoid, InstanceNorm2d
import torch
import numpy as np
import random




class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1):
        super().__init__()
        if stride == 1:
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding='same', padding_mode='reflect', bias=False)
        else:
            self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=stride, padding=int(kernel / 2), padding_mode='reflect', bias=False)
        self.relu = LeakyReLU(0.2)
        self.bn = InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x




class SkipConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, stride=1, padding='same', padding_mode='reflect', bias=False)
        self.bn = InstanceNorm2d(out_channels)
        self.relu = LeakyReLU(0.2)

    def forward(self, x, skip):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + skip)
        return x



class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, 3)
        self.skipconv = SkipConv2D(out_channels, out_channels, 3)

    
    def forward(self, x):
        c = self.conv(x)
        x = self.skipconv(x, c)
        return x



class Discriminator(Module):
    architecture = 'Atlas'
    def __init__(self, filters=64, residuals=1, drop=0.5, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.drop = drop
        self.dp = Dropout2d(self.drop)
        self.conv1 = Conv2D(2, filters, 7, 1)
        self.rb1 = Sequential(*[ResidualBlock(filters, filters) for _ in range(residuals)])
        self.sc1 = SkipConv2D(filters, filters, 3)
        self.conv2 = Conv2D(filters, filters * 2, 5, 2)
        self.rb2 = Sequential(*[ResidualBlock(filters * 2, filters * 2) for _ in range(residuals)])
        self.sc2 = SkipConv2D(filters * 2, filters * 2, 3)
        self.conv3 = Conv2D(filters * 2, filters * 4, 3, 2)
        self.rb3 = Sequential(*[ResidualBlock(filters * 4, filters * 4) for _ in range(residuals)])
        self.sc3 = SkipConv2D(filters * 4, filters * 4, 3)
        self.conv4 = Conv2D(filters * 4, filters * 6, 3, 2)
        self.rb4 = Sequential(*[ResidualBlock(filters * 6, filters * 6) for _ in range(residuals)])
        self.sc4 = SkipConv2D(filters * 6, filters * 6, 3)
        self.conv5 = Conv2D(filters * 6, filters * 8, 3, 2)
        self.rb5 = Sequential(*[ResidualBlock(filters * 8, filters * 8) for _ in range(residuals)])
        self.sc5 = SkipConv2D(filters * 8, filters * 8, 3)
        self.outconv = Conv2d(filters * 8, 8, kernel_size=self.img_size // 16, stride=2, padding=0)
        self.sigmoid = Sigmoid()


    def forward(self, x):
        c1 = self.conv1(x)
        x = self.rb1(c1)
        x = self.sc1(x, c1)
        x = self.dp(x)
        c2 = self.conv2(x)
        x = self.rb2(c2)
        x = self.sc2(x, c2)
        x = self.dp(x)
        c3 = self.conv3(x)
        x = self.rb3(c3)
        x = self.sc3(x, c3)
        x = self.dp(x)
        c4 = self.conv4(x)
        x = self.rb4(c4)
        x = self.sc4(x, c4)
        x = self.dp(x)
        c5 = self.conv5(x)
        x = self.rb5(c5)
        x = self.sc5(x, c5)
        x = self.outconv(x)
        x = self.sigmoid(x)
        return x
    



def atlas_strong(drop, img_size):
    return Discriminator(64, 1, drop, img_size)


def atlas_medium(drop, img_size):
    return Discriminator(48, 1, drop, img_size)


def atlas_light(drop, img_size):
    return Discriminator(24, 2, drop, img_size)


def atlas_ultralight(drop, img_size):
    return Discriminator(24, 1, drop, img_size)




class Model:
    def __init__(self, model_path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

        self.model_path = model_path

        self.img_channels = int(self.model_path.split('_')[3][0])
        self.img_size = int(self.model_path.split('_')[1])
        self.img_mode = self.model_path.split('_')[3]
        self.labels = int(self.model_path.split('_')[4][0])

        self.device = device
        self.model = atlas_light(0, self.img_size).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device)) 
        self.model.eval()


    def predict(self, input: torch.Tensor, threshold=0.5) -> np.array:
        with torch.no_grad():
            prediction = self.model(input.to(self.device))
        if threshold is not None:
            prediction = np.uint8(prediction.detach().cpu().numpy().squeeze() > threshold)
        else:
            prediction = prediction.detach().cpu().numpy().squeeze()
        return prediction


# print(torch.cuda.is_available())