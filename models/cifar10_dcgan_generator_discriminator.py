import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# # number of gpu's available
# nc = 3
# ngpu = 1
# # input noise dimension
# nz = 100
# # number of generator filters
# ngf = 64
# # number of discriminator filters
# ndf = 64


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator_cifar10(nn.Module):
    def __init__(self, args):
        super(Generator_cifar10, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(args.ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.nz, args.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            return output


# netG = Generator(ngpu).cuda()
# netG.apply(weights_init)
# # load weights to test the model
# # netG.load_state_dict(torch.load('weights/netG_epoch_24.pth'))
# print(netG)


class Discriminator_cifar10(nn.Module):
    def __init__(self, args):
        super(Discriminator_cifar10, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8   (ndf*4) x 4 x 4
            # nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(args.ndf * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


# netD = Discriminator(ngpu).cuda()
# netD.apply(weights_init)
# # load weights to test the model
# # netD.load_state_dict(torch.load('weights/netD_epoch_24.pth'))
# print(netD)

class Generator_mnist(nn.Module):
    def __init__(self, args):
        super(Generator_mnist, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(args.nz, args.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.network(input)
        return output


class Discriminator_mnist(nn.Module):
    def __init__(self, args):
        super(Discriminator_mnist, self).__init__()
        self.network = nn.Sequential(

            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 2, args.ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(1)
