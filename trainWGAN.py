import os
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np

from DataSetFace import DataSetFace
from wgan import Generator, Discriminator, weights_init

root = "./face/xinggan_face"
dataset = DataSetFace(root)
print(len(dataset))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

net_G = Generator(ngpu=1, nz=100, ngf=64).cuda()
net_D = Discriminator(1, nc=3, ndf=64).cuda()

lr = 5e-5
opt_G = optim.RMSprop(net_G.parameters(), lr=lr)
opt_D = optim.RMSprop(net_D.parameters(), lr=lr)

num_epochs = 30

D_losses, G_losses = [], []
iter = 0
for epoch in range(num_epochs):
    for batch_id, data in enumerate(dataloader):
        net_D.zero_grad()
        real = data.cuda()
        real_loss = net_D(real)
        noise = torch.randn(128, 100, 1, 1).cuda()
        fake = net_G(noise)
        fake_loss = net_D(fake.detach())

        D_loss = -torch.mean(real_loss) + torch.mean(fake_loss)
        D_loss.backward()
        opt_D.step()

        for p in net_D.parameters():
            p.data.clamp_(-0.01, 0.01)

        if batch_id % 5 == 0:
            net_G.zero_grad()

            g_loss = -torch.mean(net_D(fake))
            g_loss.backward()
            opt_G.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                  % (epoch, num_epochs, batch_id, len(dataloader), D_loss.item(), g_loss.item()))
            iter += 1
            D_losses.append(D_loss.item())
            G_losses.append(g_loss.item())

        if iter % 100 == 0:
            with torch.no_grad():
                noise = torch.randn(128, 100, 1, 1).cuda()
                fake = net_G(noise).detach().cpu()

                img_batch = vutils.make_grid(fake[:64], padding=2, normalize=False)
                img_batch = np.transpose(img_batch, (1, 2, 0)).numpy()
                img_batch = img_batch[:, :, :] * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]

                plt.figure(figsize=(8, 8))
                plt.axis("off")
                plt.title("Generate Images")
                plt.imshow(img_batch)
                model_path = "output/wgan/" + str(iter)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                plt.savefig(model_path + "/fake.png")
                torch.save(net_G.state_dict(), model_path + "/model.pth")

x = range(iter)
plt.figure(figsize=(10, 5))
plt.title("WGAN Loss Training img")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.plot(x, D_losses, '-', label="D")
plt.plot(x, G_losses, '-', label="G")
plt.legend()
plt.grid(True)
plt.savefig("output/wgan/loss.png")
