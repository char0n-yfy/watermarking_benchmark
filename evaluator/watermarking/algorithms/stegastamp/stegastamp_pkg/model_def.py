import torch
import torch.nn as nn
import torch.nn.functional as F


class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', kernel_initializer='he_normal'):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if kernel_initializer == 'he_normal':
            nn.init.kaiming_normal_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation is None:
            pass
        else:
            raise NotImplementedError
        return x


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation='relu', strides=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, strides, int((kernel_size - 1) / 2))
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.activation == 'relu':
            x = F.relu(x, inplace=True)
        elif self.activation is None:
            pass
        else:
            raise NotImplementedError
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


class StegaStampEncoder(nn.Module):
    def __init__(self, secret_size: int = 100):
        super().__init__()
        self.secret_dense = Dense(secret_size, 7500, activation='relu', kernel_initializer='he_normal')

        self.conv1 = Conv2D(6, 32, 3, activation='relu')
        self.conv2 = Conv2D(32, 32, 3, activation='relu', strides=2)
        self.conv3 = Conv2D(32, 64, 3, activation='relu', strides=2)
        self.conv4 = Conv2D(64, 128, 3, activation='relu', strides=2)
        self.conv5 = Conv2D(128, 256, 3, activation='relu', strides=2)
        self.up6 = Conv2D(256, 128, 3, activation='relu')
        self.conv6 = Conv2D(256, 128, 3, activation='relu')
        self.up7 = Conv2D(128, 64, 3, activation='relu')
        self.conv7 = Conv2D(128, 64, 3, activation='relu')
        self.up8 = Conv2D(64, 32, 3, activation='relu')
        self.conv8 = Conv2D(64, 32, 3, activation='relu')
        self.up9 = Conv2D(32, 32, 3, activation='relu')
        self.conv9 = Conv2D(70, 32, 3, activation='relu')
        self.residual = Conv2D(32, 3, 1, activation=None)

    def forward(self, inputs):
        secret, image = inputs
        secret = secret - 0.5
        image = image - 0.5

        secret = self.secret_dense(secret)
        secret = secret.reshape(-1, 3, 50, 50)
        secret_enlarged = F.interpolate(secret, scale_factor=(8, 8), mode='nearest')

        x = torch.cat([secret_enlarged, image], dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        up6 = self.up6(F.interpolate(conv5, scale_factor=(2, 2)))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = self.conv6(merge6)
        up7 = self.up7(F.interpolate(conv6, scale_factor=(2, 2)))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        up8 = self.up8(F.interpolate(conv7, scale_factor=(2, 2)))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        up9 = self.up9(F.interpolate(conv8, scale_factor=(2, 2)))
        merge9 = torch.cat([conv1, up9, x], dim=1)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual


class SpatialTransformerNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(320000, 128, activation='relu'),
            nn.Linear(128, 6)
        )
        with torch.no_grad():
            self.localization[-1].weight.data.zero_()
            self.localization[-1].bias.data = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32)

    def forward(self, image):
        theta = self.localization(image)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        transformed_image = F.grid_sample(image, grid, align_corners=False)
        return transformed_image


class StegaStampDecoder(nn.Module):
    def __init__(self, secret_size: int = 100):
        super().__init__()
        self.secret_size = secret_size
        self.stn = SpatialTransformerNetwork()
        self.decoder = nn.Sequential(
            Conv2D(3, 32, 3, strides=2, activation='relu'),
            Conv2D(32, 32, 3, activation='relu'),
            Conv2D(32, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 64, 3, activation='relu'),
            Conv2D(64, 64, 3, strides=2, activation='relu'),
            Conv2D(64, 128, 3, strides=2, activation='relu'),
            Conv2D(128, 128, 3, strides=2, activation='relu'),
            Flatten(),
            Dense(21632, 512, activation='relu'),
            Dense(512, secret_size, activation=None)
        )

    def forward(self, image):
        image = image - 0.5
        transformed = self.stn(image)
        return torch.sigmoid(self.decoder(transformed))

