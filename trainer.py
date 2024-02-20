import os
from argparse import ArgumentParser
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from generator import Generator
from discriminator import Discriminator
import torchvision.utils as vutils
from dataset import face_loader, invTrans
import matplotlib.pyplot as plt


logger = SummaryWriter('./log')
torch.random.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)
device = "cuda"
epochs = 2
verbose_step = 25
save_step = 100
beta = 0.5
init_lr = 0.0002
z_dim = 100

G_losses = []
D_losses = []


def save_checkpoint(model_, epoch_, optm, checkpoint_path):
    save_dict = {
        'epoch': epoch_,
        'model_state_dict': model_.state_dict(),
        'optimizer_state_dict': optm.state_dict()
    }
    torch.save(save_dict, checkpoint_path)

def train():
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        '--c', # G and D checkpoint path: model_g_xxx.pth~model_d_xxx.pth
        default=None,
        type=str,
        help='training from scratch or resume training'
    )
    args = parser.parse_args()

    # model init
    G = Generator() # new a generator model instance
    G.apply(G.weights_init) # apply weight init for G
    D = Discriminator()  # new a discriminator model instance
    D.apply(D.weights_init)  # apply weight init for G
    # gpu训练
    G.to(device)
    D.to(device)

    # loss criterion
    criterion = nn.BCELoss() # binary classification loss

    # 优化方法
    g_optimizer = optim.Adam(G.parameters(), lr=init_lr, betas=(beta, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=init_lr, betas=(beta, 0.999))

    start_epoch, step = 0, 0 # start position

    if args.c:
        model_g_path = args.c.split('~')[0]
        checkpoint_g = torch.load(model_g_path)
        G.load_state_dict(checkpoint_g['model_state_dict'])
        g_optimizer.load_state_dict(checkpoint_g['optimizer_state_dict'])
        start_epoch_gc = checkpoint_g['epoch']

        model_d_path = args.c.split('~')[1]
        checkpoint_d = torch.load(model_d_path)
        D.load_state_dict(checkpoint_d['model_state_dict'])
        d_optimizer.load_state_dict(checkpoint_d['optimizer_state_dict'])
        start_epoch_dc = checkpoint_d['epoch']

        start_epoch = start_epoch_gc if start_epoch_dc > start_epoch_gc else start_epoch_dc
        print('Resume Training From Epoch: %d' % start_epoch)
    else:
        print('Training From Scratch!')

    G.train()
    D.train()

    flz = torch.randn(size=(64, 100), device=device)

    # 循环开始
    for epoch in range(start_epoch, epochs):
        print('Start Epoch: %d, Steps: %d' % (epoch, len(face_loader)))
        for batch, _ in face_loader:
            b_size = batch.size(0) # 64

            # 处理 Discriminator
            # 初始化
            d_optimizer.zero_grad()

            labels_gt = torch.full(size=(b_size, ), fill_value=0.9, dtype=torch.float, device=device)
            predict_labels_gt = D(batch.to(device)).squeeze()
            loss_d_of_gt = criterion(predict_labels_gt, labels_gt)

            labels_fake = torch.full(size=(b_size, ), fill_value=0.1, dtype=torch.float, device=device)
            latent_z = torch.randn(size=(b_size, z_dim), device=device)
            predict_labels_fake = D(G(latent_z)).squeeze()
            loss_d_of_fake = criterion(predict_labels_fake, labels_fake)

            loss_D = loss_d_of_gt + loss_d_of_fake
            loss_D.backward()
            d_optimizer.step()
            logger.add_scalar('Loss/Discriminator', loss_D.mean().item(), step)

            # 处理 Generator
            # 初始化
            g_optimizer.zero_grad()
            latent_z = torch.randn(size=(b_size, z_dim), device=device)
            labels_for_g = torch.full(size=(b_size, ), fill_value=0.9, dtype=torch.float, device=device)
            predict_labels_from_g = D(G(latent_z)).squeeze() # [N, ]

            loss_G = criterion(predict_labels_from_g, labels_for_g)
            loss_G.backward()
            g_optimizer.step()
            logger.add_scalar('Loss/Generator', loss_G.mean().item(), step)

            if not step % verbose_step:
                with torch.no_grad():
                    fake_image_dev = G(flz)
                    logger.add_image('Generator Faces', invTrans(vutils.make_grid(fake_image_dev.detach().cpu(), nrow=8)), step)

            if not step % save_step: # save G and D
                print('model_g_%d_%d.pth' % (epoch, step))
                model_path = 'model_g_%d_%d.pth' % (epoch, step)
                save_checkpoint(G, epoch,g_optimizer, os.path.join('model_save', model_path))
                model_path = 'model_d_%d_%d.pth' % (epoch, step)
                save_checkpoint(D, epoch, d_optimizer, os.path.join('model_save', model_path))

                # 保存图片
                vutils.save_image(G(latent_z).data[:64], os.path.join('output', 'fake_epoch_%d_%d.png' % (epoch, step)), nrow=8,
                           normalize=True)

            step += 1
            logger.flush()
            print('Epoch: [%d/%d], step: %d G loss: %.3f, D loss %.3f' %
                  (epoch, epochs, step, loss_G.mean().item(), loss_D.mean().item()))

            # loss数据处理
            G_losses.append(loss_G.mean().item())
            D_losses.append(loss_D.mean().item())
    logger.close()


if __name__ == '__main__':
    train()

    # 显示loss曲线
    plt.figure(figsize=(10, 5))
    plt.title("G&D Loss Training img")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()




