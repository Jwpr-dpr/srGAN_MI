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

def training_step( generator, discriminator, gen_optimizer, disc_optimizer, real_hr, lr_input,
                bce_loss, perceptual_loss, adv_weight=1e-3, content_weight=1.0, device=device):
    
    generator.train()
    discriminator.train()

    real_hr = real_hr.to(device)
    lr_input = lr_input.to(device)
    batch_size = real_hr.size(0)

    real_labels = torch.ones((batch_size, 1), device=device)
    fake_labels = torch.zeros((batch_size, 1), device=device)

    # Train Discriminator
    disc_optimizer.zero_grad()
    with torch.no_grad():
        fake_hr = generator(lr_input)
    pred_real = discriminator(real_hr)
    pred_fake = discriminator(fake_hr)
    d_loss_real = bce_loss(pred_real, real_labels)
    d_loss_fake = bce_loss(pred_fake, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    disc_optimizer.step()

    # Train Generator
    gen_optimizer.zero_grad()
    fake_hr = generator(lr_input)
    pred_fake = discriminator(fake_hr)
    adversarial_loss = bce_loss(pred_fake, real_labels)
    content_loss = perceptual_loss(fake_hr, real_hr)
    g_loss = content_weight * content_loss + adv_weight * adversarial_loss
    g_loss.backward()
    gen_optimizer.step()

    return {
        'g_loss': g_loss.item(),
        'd_loss': d_loss.item(),
        'adv_loss': adversarial_loss.item(),
        'content_loss': content_loss.item()
    }
def train(pathToData:str, pathToTrainFile:str, pathToTestFile:str, pathToValFile:str,
          bathSize:int, training:str,learningRate:float, epochs:int, generator_path:str):
    
    if training == 'tuning':
        generator.load_state_dict(torch.load(generator_path))
    
    generator.to(device)
    generator.train()
    return