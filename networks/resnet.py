import torch.nn as nn


class ResidualNetwork(nn.Module):

    def __init__(self, in_features, target_features, width, depth, final_time, activation=nn.Tanh(), bias=True,
                 opening_layer=True, closing_layer=True):
        super(ResidualNetwork, self).__init__()

        self.in_features = in_features
        self.width = width
        self.depth = depth
        self.final_time = final_time

        self.opening_layer = None
        if opening_layer:
            self.opening_layer = nn.Linear(in_features, width, bias=bias)

        self.residual_layer = ResidualLayer(width, depth, final_time, activation=activation, bias=bias)

        self.closing_layer = None
        if closing_layer:
            self.closing_layer = nn.Linear(width, target_features, bias=bias)

    def forward(self, x):

        if self.opening_layer is not None:
            x = self.opening_layer(x)

        x = self.residual_layer(x)

        if self.closing_layer is not None:
            x = self.closing_layer(x)

        return x


class ResidualLayer(nn.Module):

    def __init__(self, width, depth, final_time, activation=nn.Tanh(), bias=True):
        super(ResidualLayer, self).__init__()
        self.width = width
        self.depth = depth
        self.final_time = final_time
        self.h = depth / final_time
        self.activation = activation

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(width, width, bias=bias))

    def forward(self, x):

        for layer in self.layers:
            dx = layer(x)
            if self.activation is not None:
                dx = self.activation(dx)
            x = x + self.h * dx

        return x





