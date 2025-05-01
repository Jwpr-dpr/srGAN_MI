import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

def generateSRimage(path_to_image, path_to_save):

    imageData = Image.open(path_to_image).convert('RGB')
    return