import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
from model import Generator, Discriminator
from load_data import CelebDataset
from torchvision import transforms
from tensorboardX import SummaryWriter


dataroot = '/home/ujjawal/Downloads/celeba' #location of the directory containing the dataset
workers = 2
batch_size = 16
epochs = 4
lr_D = 0.00015  #learning rate for discriminator
lr_G = 0.00015  #learning rate for generator

#transforming the data i.e. resizing the shape to 64*64, normalizing to mean=0 and stddev=0.5, transforming pil image to tensor
transform = transforms.Compose([transforms.Resize(64), transforms.CenterCrop(64), transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#creating an object of class CelebDataset
CelebDataProcessed = CelebDataset(dataroot, batch_size, workers, transform)
#fetching the data dictionary returned by get_loader() function of our class instance
data_dict = CelebDataProcessed.get_loader()
processed_dataset = data_dict['dataset']
dataloader = data_dict['data_loader']

#creating instance of Generator() as well as Discriminator() class
netG = Generator()
netD = Discriminator()

#loss used is binary cross-entropy loss
criterion = nn.BCELoss()

#parameterizing real_labels as 1 and fake_labels as 0
real_label = 1
fake_label = 0

#fixed noise to keep tab of images generated via generator
fixed_noise = torch.randn(64, 100, 1, 1)

#optimizers for netG() and netD()
optimD = optim.SGD(netD.parameters(), lr=lr_D, momentum=0.9)
optimG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.5, 0.999))

#creating a summary writer to monitor progress via tensorboard
writer = SummaryWriter('/home/ujjawal/PycharmProjects/GAN/venv/log_dir/sixth_run')

#weight initialization technique as defined in the paper
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


netG.apply(weights_init)
netD.apply(weights_init)

#train loop, 1:1 training for generator and discriminator, used trick from GAN hacks to train discriminator
for epoch in range(epochs):
    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        b_size = data[0].size(0)
        label = torch.full((b_size,), real_label)
        output = netD(data[0]).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, 100, 1, 1)
        fake = netG(noise)
        label.fill_(fake_label)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimD.step()

        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimG.step()

        if i % 500 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        writer.add_scalar('DiscriminatorLoss', errD, i)
        writer.add_scalar('GeneratorLoss', errG, i)

        #if i % 500 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach()
        writer.add_image('Generated images from fixed noise', torchvision.utils.make_grid(fake, nrow=4, normalize=True), i)

writer.close()






