import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import numpy as np 
from torchvision.transforms import InterpolationMode
# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from ImageNet
data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
v2_transform = v2.Compose([
            v2.RandomResizedCrop((224,224)),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(degrees=30),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
            ])

