import torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        # shape is a tuple
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class MNISTAutoencoder(nn.Module):

    def __init__(self, width=16, intrinsic_dim=50):
        super(MNISTAutoencoder,self).__init__()

        enc1 = nn.Conv2d(in_channels=1, out_channels=width,
                              kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        enc2 = nn.Conv2d(in_channels=width, out_channels=2 * width,
                              kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        act2 = nn.ReLU()
        encr = View((-1, 7 * 7 * 2 * width,))
        enc3 = nn.Linear(in_features=7 * 7 * 2 * width, out_features=intrinsic_dim, bias=True)

        self.enc = nn.Sequential(enc1, act1, enc2, act2, encr, enc3)

        dec0 = nn.Linear(in_features=intrinsic_dim, out_features=7 * 7 * 2 * width, bias=True)
        decr = View((-1, width * 2, 7, 7))
        decb1 = nn.BatchNorm2d(width * 2)
        dec1 = nn.ConvTranspose2d(in_channels=2 * width, out_channels=width,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        decb2 = nn.BatchNorm2d(width)
        dec2 = nn.ConvTranspose2d(in_channels=width, out_channels=1,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        self.dec = nn.Sequential(dec0, decr, decb1, dec1, act1, decb2, dec2)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class MNISTAutoencoderFeatureExtractor(nn.Module):

    def __init__(self, width=16, intrinsic_dim=50):
        super(MNISTAutoencoderFeatureExtractor,self).__init__()

        enc1 = nn.Conv2d(in_channels=1, out_channels=width,
                              kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        enc2 = nn.Conv2d(in_channels=width, out_channels=2 * width,
                              kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)

        act2 = nn.ReLU()
        encr = View((-1, 7 * 7 * 2 * width,))
        enc3 = nn.Linear(in_features=7 * 7 * 2 * width, out_features=intrinsic_dim, bias=True)

        self.enc = nn.Sequential(enc1, act1, enc2, act2, encr, enc3)

        dec0 = nn.Linear(in_features=intrinsic_dim, out_features=7 * 7 * 2 * width, bias=True)
        decr = View((-1, width * 2, 7, 7))
        decb1 = nn.BatchNorm2d(width * 2)
        dec1 = nn.ConvTranspose2d(in_channels=2 * width, out_channels=width,
                                  kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=True)
        act1 = nn.ReLU()
        decb2 = nn.BatchNorm2d(width)

        self.dec = nn.Sequential(dec0, decr, decb1, dec1, act1, decb2)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

