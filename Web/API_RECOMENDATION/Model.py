from pydantic import BaseModel
from PIL import Image
import torch
import timm
import torch.nn as nn
from torchvision import models, transforms
import cv2
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#detect-fashion
path_fashion_model = r"D:\GITHUB\Vouge\Model-Fashion\best.pt"
fashion_model = YOLO(path_fashion_model)
fashion_model = fashion_model.to(device)

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
model_gen = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")
model_gen = model_gen.to(device)

# Load model person detection
person_model=YOLO("yolov8n.pt")
person_model=person_model.to(device)

# Features Extraction of DINOv2
path_label_model = r"D:\GITHUB\VITON_Project\Virtual_Try_On\Features_DataBase\labels_vit_large_patch14_dinov2.npy"
path_saved_features = r"D:\GITHUB\VITON_Project\Virtual_Try_On\Features_DataBase\features_vit_large_patch14_dinov2.npy"


# Load pre-trained model DINOv2
class DINOv2ViT(nn.Module):
    def __init__(self, model_name="vit_large_patch14_dinov2", img_size=None):
        super().__init__()
        if img_size is None:
            self.vit = timm.create_model(model_name, pretrained=True)
            self.img_size = self.vit.default_cfg["input_size"][-1]
        else:
            self.vit = timm.create_model(model_name, pretrained=True, img_size=img_size)
            self.img_size = img_size

        self.out_dim = self.vit.embed_dim
        self.mean = self.vit.default_cfg["mean"]
        self.std = self.vit.default_cfg["std"]

    def forward(self, x):
        embedding = self.vit(x)

        return embedding

# Model DinoV2
model = DINOv2ViT()
model = model.to(device)
model.eval()


#detect-fashion
fashion_model = YOLO(path_fashion_model)
fashion_model = fashion_model.to(device)