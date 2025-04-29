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
from utils import * 
from tqdm import tqdm


class SR_Trainer:
    def __init__(self,_res_num, _scale,_path_size):
        self.res_num = _res_num
        self.scale = _scale
        self.patch_size = _path_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bce_loss = nn.BCELoss()
        self.pixel_loss = nn.MSELoss()
        self.perceptual_loss = vgg19()
        self.generator = Generator(3, 64, 3).to(self.device)
        self.discriminator = Discriminator(self.patch_size * self.scale).to(self.device)
        self.date = datetime.today().strftime('%d-%m-%Y')
        self.save_model_path = f"../models/SRGAN-{self.date}.pth"
        self.writer = SummaryWriter(f"../logs/runs/SRGAN-training-{self.date}")
        
        pass
    
    cudnn.benchmark = True

    def train(self, GT_path: str, LR_path: str, val_GT_path: str, val_LR_path: str, 
            batch_size: int, epochs: int, process: str, generator_path: str,
            feat_layer, vgg_rescale_coeff, adv_coeff, tv_loss_coeff,
            scale: int = 4, num_workers: int = 3):

        init_val_loss = 1e10
        init_val_psnr = 0

        transform = transforms.Compose([CropPatch(scale, self.patch_size), Augmentation()])
        
        train_set = SuperResolutionDataset(GT_path, LR_path, in_memory=False, transform=transform)
        val_set = SuperResolutionDataset(val_GT_path, val_LR_path, in_memory=False, transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        g_optim = optim.Adam(self.generator.parameters(), lr=1e-4)
        d_optim = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        
        scheduler = optim.lr_scheduler.StepLR(g_optim, step_size=2000, gamma=0.1)

        l2_loss = nn.MSELoss()
        
        if process == "tunning":
            self.generator.load_state_dict(torch.load(generator_path))
        self.generator.train()
        self.discriminator.train()
        
        tv_loss = TVLoss()

        for epoch in tqdm(range(epochs), desc="Entrenamiento por épocas"):
            
            args = type('', (), {})()
            args.feat_layer = feat_layer
            args.vgg_rescale_coeff = vgg_rescale_coeff
            args.adv_coeff = adv_coeff
            args.tv_loss_coeff = tv_loss_coeff
            
            self.epoch_train(self.generator, train_loader, g_optim, l2_loss, self.device, self.discriminator, d_optim, self.perceptual_loss, self.bce_loss, tv_loss, args)
            metrics = self.evaluate_epoch(self.generator, val_loader, self.device, l2_loss, scale)
            scheduler.step()
            
            tqdm.write(f"[Training epoch: {epoch}] Val Loss: {metrics['val_loss']:.4f}, Val PSNR: {metrics['val_psnr']:.2f}")
            tqdm.set_postfix(val_loss=metrics['val_loss'], val_psnr=['val_psnr'])

            if (epoch % 500 == 0) or (metrics['val_loss'] < init_val_loss and metrics['val_psnr'] > init_val_psnr):
                torch.save(self.generator.state_dict(), f'./model/SRGAN_gene_{epoch:03d}.pt')
                torch.save(self.discriminator.state_dict(), f'./model/SRGAN_discrim_{epoch:03d}.pt')
                init_val_psnr = metrics['val_psnr']
                init_val_loss = metrics['val_loss']

    
    def epoch_train(
        generator: nn.Module, loader, g_optim, l2_loss, device,
        discriminator, d_optim, VGG_loss, cross_ent, tv_loss, args
    ):
        generator.train()
        discriminator.train()
        
        for data in loader:
            gt = data['GT'].to(device)
            lr = data['LR'].to(device)
            
            # --- Train Discriminator ---
            with torch.no_grad():
                output, _ = generator(lr)
            fake_prob = discriminator(output)
            real_prob = discriminator(gt)
            
            d_loss_real = cross_ent(real_prob, torch.ones_like(real_prob))
            d_loss_fake = cross_ent(fake_prob, torch.zeros_like(fake_prob))
            d_loss = d_loss_real + d_loss_fake
            
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()

            # --- Train Generator ---
            output, _ = generator(lr)
            fake_prob = discriminator(output)
            
            _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1) / 2, (output + 1) / 2, layer=args.feat_layer)

            l2 = l2_loss(output, gt)
            percep = args.vgg_rescale_coeff * _percep_loss
            adv = args.adv_coeff * cross_ent(fake_prob, torch.ones_like(fake_prob))
            tv = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

            total_gen_loss = l2 + percep + adv + tv

            g_optim.zero_grad()
            total_gen_loss.backward()
            g_optim.step()


    def evaluate_epoch(generator, loader, device, l2_loss, scale, VGG_loss, cross_ent, discriminator, args,tv_loss):
        generator.eval()
        discriminator.eval()

        total_loss = 0
        total_percep_loss = 0
        total_adv_loss = 0
        total_tv_loss = 0
        psnr_list = []

        with torch.no_grad():
            for data in loader:
                gt = data['GT'].to(device)
                lr = data['LR'].to(device)

                output, _ = generator(lr)
                fake_prob = discriminator(output)

                _percep_loss, hr_feat, sr_feat = VGG_loss((gt + 1) / 2, (output + 1) / 2, layer=args.feat_layer)

                l2 = l2_loss(output, gt)
                percep = args.vgg_rescale_coeff * _percep_loss
                adv = args.adv_coeff * cross_ent(fake_prob, torch.ones_like(fake_prob))
                tv = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

                total_batch_loss = l2 + percep + adv + tv

                total_loss += total_batch_loss.item()
                total_percep_loss += percep.item()
                total_adv_loss += adv.item()
                total_tv_loss += tv.item()

                # PSNR
                output = (output[0].cpu().numpy() + 1) / 2
                gt = (gt[0].cpu().numpy() + 1) / 2
                output = np.clip(output, 0, 1).transpose(1, 2, 0)
                gt = gt.transpose(1, 2, 0)

                y_output = rgb2ycbcr(output)[scale:-scale, scale:-scale, :1]
                y_gt = rgb2ycbcr(gt)[scale:-scale, scale:-scale, :1]

                psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range=1.0)
                psnr_list.append(psnr)

        metrics = {
        'avg_loss' : total_loss / len(loader),
        'avg_percep_loss' : total_percep_loss / len(loader),
        'avg_adv_loss' : total_adv_loss / len(loader),
        'avg_tv_loss' : total_tv_loss / len(loader),
        'avg_psnr' : np.mean(psnr_list)
        }
        return metrics

    def test(GT_path, LR_path, res_num, generator_path, scale, num_workers):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = SuperResolutionDataset(GT_path, LR_path, False, None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

        generator = Generator(3, 64, 3, res_num).to(device)
        generator.load_state_dict(torch.load(generator_path))
        generator.eval()

        f = open('./result.txt', 'w')
        psnr_list = []
        cycle_loss_list = []

        with torch.no_grad():
            for i, data in enumerate(loader):
                gt = data['GT'].to(device)
                lr = data['LR'].to(device)
                output, _ = generator(lr)

                # Compute PSNR
                output_img = (output[0].cpu().numpy() + 1) / 2
                gt_img = (gt[0].cpu().numpy() + 1) / 2
                output_img = np.clip(output_img, 0, 1).transpose(1, 2, 0)
                gt_img = gt_img.transpose(1, 2, 0)
                y_output = rgb2ycbcr(output_img)[scale:-scale, scale:-scale, :1]
                y_gt = rgb2ycbcr(gt_img)[scale:-scale, scale:-scale, :1]

                psnr = compare_psnr(y_output / 255.0, y_gt / 255.0, data_range=1.0)
                psnr_list.append(psnr)

                # Compute Cycle Consistency Loss
                # Note: generator expects input batch, not a single image
                rec_loss = cycle_consistency_loss(
                    lambda x: generator(x)[0],  # extract only the output tensor
                    lr,
                    downsample_fn=lambda x, sf=scale: downsample(x, scale_factor=sf),
                    scale_factor=scale
                )
                cycle_loss_list.append(rec_loss)

                # Save the image
                Image.fromarray((output_img * 255.0).astype(np.uint8)).save(f'./result/res_{i:04d}.png')

                # Log both metrics per image
                f.write(f'Image {i:04d} | PSNR: {psnr:.4f} | Cycle Loss: {rec_loss:.6f}\n')

            # Average metrics
            avg_psnr = np.mean(psnr_list)
            avg_cycle_loss = np.mean(cycle_loss_list)

            f.write(f'\nAverage PSNR: {avg_psnr:.4f}\n')
            f.write(f'Average Cycle Consistency Loss: {avg_cycle_loss:.6f}\n')
        f.close()