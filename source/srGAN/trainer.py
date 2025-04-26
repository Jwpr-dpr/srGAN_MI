import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from losses import TVLoss, perceptual_loss
from srgan_model import Generator, Discriminator
from vgg19 import vgg19
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from skimage.measure import compare_psnr
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from torch.backends import cudnn
import torch.nn.functional as F
from dataset import * 

date = datetime.today().strftime('%d-%m-%Y')
save_model_path = f"../models/SRGAN-{date}.pth"
writer = SummaryWriter(f"../logs/runs/SRGAN-training-{date}")
cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bce_loss = nn.BCELoss()
pixel_loss = nn.MSELoss()
perceptual_loss = vgg19()
generator = Generator()
discriminator = Discriminator()


def train(
    GT_path, LR_path, val_GT_path, val_LR_path, in_memory,
    batch_size, num_workers, patch_size, scale, res_num,
    pre_train_epoch, fine_train_epoch, fine_tuning, generator_path,
    feat_layer, vgg_rescale_coeff, adv_coeff, tv_loss_coeff
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([CropPatch(scale, patch_size), Augmentation()])
    train_set = SuperResolutionDataset(GT_path, LR_path, in_memory, transform)
    val_set = SuperResolutionDataset(val_GT_path, val_LR_path, in_memory, transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    generator = Generator(3, 64, 3, res_num, scale).to(device)
    if fine_tuning:
        generator.load_state_dict(torch.load(generator_path))
    generator.train()

    l2_loss = nn.MSELoss()
    g_optim = optim.Adam(generator.parameters(), lr=1e-4)

    for epoch in range(pre_train_epoch):
        epoch_train(generator, train_loader, g_optim, l2_loss, device, phase='pre')
        if epoch % 2 == 0:
            print(f"[Pre-train Epoch {epoch}] done")
        if epoch % 800 == 0:
            torch.save(generator.state_dict(), f'./model/pre_trained_model_{epoch:03d}.pt')

    discriminator = Discriminator(patch_size * scale).to(device)
    discriminator.train()
    d_optim = optim.Adam(discriminator.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

    vgg_net = vgg19().to(device).eval()
    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    tv_loss = TVLoss()

    for epoch in range(fine_train_epoch):
        scheduler.step()
        args = type('', (), {})()
        args.feat_layer = feat_layer
        args.vgg_rescale_coeff = vgg_rescale_coeff
        args.adv_coeff = adv_coeff
        args.tv_loss_coeff = tv_loss_coeff
        epoch_train(generator, train_loader, g_optim, l2_loss, device, discriminator, d_optim, VGG_loss, cross_ent, tv_loss, args, phase='fine')

        val_loss, val_psnr = evaluate_epoch(generator, val_loader, device, l2_loss, scale)
        print(f"[Fine-tune Epoch {epoch}] Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}")

        if epoch % 500 == 0:
            torch.save(generator.state_dict(), f'./model/SRGAN_gene_{epoch:03d}.pt')
            torch.save(discriminator.state_dict(), f'./model/SRGAN_discrim_{epoch:03d}.pt')

def epoch_train(generator, loader, g_optim, l2_loss, device, discriminator=None, d_optim=None, VGG_loss=None, cross_ent=None, tv_loss=None, args=None, phase='pre'):
    generator.train()
    for data in loader:
        gt = data['GT'].to(device)
        lr = data['LR'].to(device)
        if phase == 'pre':
            output, _ = generator(lr)
            loss = l2_loss(gt, output)
            g_optim.zero_grad()
            loss.backward()
            g_optim.step()
        else:
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)

            d_loss_real = cross_ent(real_prob, torch.ones_like(real_prob))
            d_loss_fake = cross_ent(fake_prob, torch.zeros_like(fake_prob))
            d_loss = d_loss_real + d_loss_fake
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            output, _ = generator(lr)
            fake_prob = discriminator(output)
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt+1)/2, (output+1)/2, layer=args.feat_layer)

            L2 = l2_loss(output, gt)
            percep = args.vgg_rescale_coeff * _percep_loss
            adv = args.adv_coeff * cross_ent(fake_prob, torch.ones_like(fake_prob))
            tv = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            g_loss = L2 + percep + adv + tv
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()

def evaluate_epoch(generator, loader, device, l2_loss, scale):
    generator.eval()
    total_loss = 0
    psnr_list = []
    with torch.no_grad():
        for data in loader:
            gt = data['GT'].to(device)
            lr = data['LR'].to(device)
            output, _ = generator(lr)
            loss = l2_loss(gt, output)
            total_loss += loss.item()

            output = (output[0].cpu().numpy() + 1) / 2
            gt = (gt[0].cpu().numpy() + 1) / 2
            output = np.clip(output, 0, 1).transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
            y_output = rgb2ycbcr(output)[scale:-scale, scale:-scale, :1]
            y_gt = rgb2ycbcr(gt)[scale:-scale, scale:-scale, :1]
            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)

    avg_loss = total_loss / len(loader)
    avg_psnr = np.mean(psnr_list)
    return avg_loss, avg_psnr

def test(GT_path, LR_path, res_num, generator_path, scale, num_workers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SuperResolutionDataset(GT_path, LR_path, False, None)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    generator = Generator(3, 64, 3, res_num).to(device)
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    f = open('./result.txt', 'w')
    psnr_list = []

    with torch.no_grad():
        for i, data in enumerate(loader):
            gt = data['GT'].to(device)
            lr = data['LR'].to(device)
            output, _ = generator(lr)

            output = (output[0].cpu().numpy() + 1) / 2
            gt = (gt[0].cpu().numpy() + 1) / 2
            output = np.clip(output, 0, 1).transpose(1, 2, 0)
            gt = gt.transpose(1, 2, 0)
            y_output = rgb2ycbcr(output)[scale:-scale, scale:-scale, :1]
            y_gt = rgb2ycbcr(gt)[scale:-scale, scale:-scale, :1]

            psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range=1.0)
            psnr_list.append(psnr)
            f.write(f'psnr : {psnr:.4f}\n')
            Image.fromarray((output * 255.0).astype(np.uint8)).save(f'./result/res_{i:04d}.png')

        f.write(f'avg psnr : {np.mean(psnr_list):.4f}')
        f.close()
