import torch.nn as nn
import torch

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Generator128(nn.Module):
    def __init__(self, config):
        super(Generator128, self).__init__()

        self.init_size = config.img_size_G // 8
        self.l1 = nn.Sequential(nn.Linear(config.latent_dim_G, 256*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(256),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, config.channels_G, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(config.channels_G, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks5(img)
        return img

class Generator256(nn.Module):
    def __init__(self, config):
        super(Generator256, self).__init__()

        self.init_size = config.img_size_G // 8
        self.l1 = nn.Sequential(nn.Linear(config.latent_dim_G, 256*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(256),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks5 = nn.Sequential(
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks6 = nn.Sequential(
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(8, config.channels_G, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(config.channels_G, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks3(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks4(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks5(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks6(img)
        return img

class Generator32(nn.Module):
    def __init__(self, config):
        super(Generator32, self).__init__()

        self.init_size = config.img_size_G // 4
        self.l1 = nn.Sequential(nn.Linear(config.latent_dim_G, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, config.channels_G, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(config.channels_G, affine=False)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(config.channels_G, 16, bn=False),
        )

        # The height and width of downsampled image
        ds_size = config.img_size_G // 2 ** 1
        self.adv_layer = nn.Sequential(nn.Linear(16 * ds_size ** 2, 1), nn.Sigmoid())

        # self.model = nn.Sequential(
        #     *discriminator_block(config.channels_G, 16, bn=False),
        #     *discriminator_block(16, 32),
        #     *discriminator_block(32, 64),
        #     *discriminator_block(64, 128),
        # )
        #
        # # The height and width of downsampled image
        # ds_size = config.img_size_G // 2 ** 4
        # self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity