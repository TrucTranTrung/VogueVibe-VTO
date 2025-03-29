import torch
import os
from collections import OrderedDict
from PIL import Image
import numpy as np
import requests
from io import BytesIO

def load_image_into_numpy_array(data):
    try:
        image = Image.open(BytesIO(data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    except Exception as e:
        print(f"Error loading image into numpy array: {e}")
        return None

label_colours = [(0,0,0)
                , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
                , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

# Get palette for mask visualization (4 classes: background + 3 classes)
def get_palette(num_classes):
    palette = [0] * (num_classes * 3)
    for i in range(num_classes):
        palette[i * 3 + 0] = i * 50
        palette[i * 3 + 1] = i * 80
        palette[i * 3 + 2] = i * 100
    return palette

# Load U-2-Net checkpoint (model)
def load_checkpoint(model, checkpoint_path):
    """Load model checkpoint for multi-GPU or single-GPU setup."""
    if not os.path.exists(checkpoint_path):
        print(f"----No checkpoints found at path: {checkpoint_path}----")
        return model
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True)
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:]  # Remove `module.` if using multi-GPU
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print(f"----Model loaded from path: {checkpoint_path}----")
    return model


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(_img):
    # _img = Image.open(img_path).convert('RGB') 
    _img = _img.resize((256, 192), Image.BILINEAR)  # (width, height)  
    return _img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample