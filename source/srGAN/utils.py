import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import resize

class _conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        
        self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
        self.bias.data = torch.zeros((out_channels))
        
        for p in self.parameters():
            p.requires_grad = True
        

class conv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
        super(conv, self).__init__()
        m = []
        m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
                               kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        
        if BN:
            m.append(nn.BatchNorm2d(num_features = out_channel))
        
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out
        
class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, act = nn.ReLU(inplace = True), bias = True):
        super(ResBlock, self).__init__()
        m = []
        m.append(conv(channels, channels, kernel_size, BN = True, act = act))
        m.append(conv(channels, channels, kernel_size, BN = True, act = None))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.body(x)
        res += x
        return res
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_res_block, act = nn.ReLU(inplace = True)):
        super(BasicBlock, self).__init__()
        m = []
        
        self.conv = conv(in_channels, out_channels, kernel_size, BN = False, act = act)
        for i in range(num_res_block):
            m.append(ResBlock(out_channels, kernel_size, act))
        
        m.append(conv(out_channels, out_channels, kernel_size, BN = True, act = None))
        
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        res = self.conv(x)
        out = self.body(res)
        out += res
        
        return out

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x):
        # Make sure input is in the form [B, C, H, W]
        if isinstance(x, torch.Tensor):
            if x.ndim == 1:  # [784]
                x = x.view(1, 1, 2048, 2048)
            elif x.ndim == 2:  # [1, 784]
                x = x.view(-1, 1, 2048, 2048)
            elif x.ndim == 3:  # [1, 28, 28]
                x = x.unsqueeze(0)
            elif x.ndim == 4:
                pass  # already [B, C, H, W]
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}")
        else:
            raise TypeError("Input must be a torch.Tensor")

        x = x.requires_grad_()

        # Forward pass
        output = self.model(x)

        # Backward pass
        self.model.zero_grad()
        output.backward(torch.ones_like(output))

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize and normalize
        cam = F.interpolate(cam, size=(2048, 2048), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam
        
class Upsampler(nn.Module):
    def __init__(self, channel, kernel_size, scale, act = nn.ReLU(inplace = True)):
        super(Upsampler, self).__init__()
        m = []
        m.append(conv(channel, channel * scale * scale, kernel_size))
        m.append(nn.PixelShuffle(scale))
    
        if act is not None:
            m.append(act)
        
        self.body = nn.Sequential(*m)
    
    def forward(self, x):
        out = self.body(x)
        return out

class discrim_block(nn.Module):
    def __init__(self, in_feats, out_feats, kernel_size, act = nn.LeakyReLU(inplace = True)):
        super(discrim_block, self).__init__()
        m = []
        m.append(conv(in_feats, out_feats, kernel_size, BN = True, act = act))
        m.append(conv(out_feats, out_feats, kernel_size, BN = True, act = act, stride = 2))
        self.body = nn.Sequential(*m)
        
    def forward(self, x):
        out = self.body(x)
        return out

def downsample(img_tensor, scale_factor=4, mode='bicubic'):
    """
    Downsamples a high-resolution image back to low-resolution using interpolation.
    """
    h, w = img_tensor.shape[-2:]
    new_h, new_w = h // scale_factor, w // scale_factor
    return resize(img_tensor, [new_h, new_w], interpolation=mode)

def denormalize(tensor, mean=0.5, std=0.5):
    return tensor * std + mean

def normalize(tensor, mean=0.5, std=0.5):
    return (tensor - mean) / std

def cycle_consistency_loss(sr_model, lr_img, downsample_fn, scale_factor=4):
    """
    Computes the L1 cycle-consistency loss between the original LR image
    and the one obtained after SR -> Downsampling.
    """
    with torch.no_grad():
        sr_img = sr_model(lr_img)                    # LR -> HR
        rec_img = downsample_fn(sr_img, scale_factor) # HR -> LR (reverse)
    
    loss = F.l1_loss(rec_img, lr_img)
    return loss.item()