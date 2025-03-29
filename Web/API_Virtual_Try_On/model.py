# -*- coding: utf-8 -*-
import sys
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from Support import get_palette, load_checkpoint
sys.path.append(r'D:\GITHUB\Virtual_try_on\Web\API_Virtual_Try_On\clothes-virtual-try-on\networks')
from u2net import U2NET  
import graph
from dataloaders import custom_transforms as tr
import deeplab_xception_transfer

# Load U-2-Net model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = r"D:\GITHUB\Vouge\Model_Mask\cloth_segm_u2net_latest.pth"

# Data transformation: Resize and normalize images
class NormalizeImage:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize = transforms.Normalize([self.mean] * 3, [self.std] * 3)

    def __call__(self, image_tensor):
        return self.normalize(image_tensor)

# Initialize and load the U-2-Net model
net = U2NET(in_ch=3, out_ch=4) 
net = load_checkpoint(net, checkpoint_path)
net = net.to(device)
net = net.eval()  


# Load Human Parsing model
loadmodel = r"D:\GITHUB\Vouge\Human_parsing\inference.pth"
use_gpu = True  
Parsing_model = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                                 hidden_layers=128,
                                                                                 source_classes=7)
Parsing_model.to(device)
# Load model
if loadmodel:
    x = torch.load(loadmodel, weights_only=True)
    Parsing_model.load_source_model(x)
else:
    print('no model load !!!!!!!!')
    raise RuntimeError('No model!!!!')
if use_gpu:
    Parsing_model.cuda()

# Get color palette for mask visualization
palette = get_palette(4)

