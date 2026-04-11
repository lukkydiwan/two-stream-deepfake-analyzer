import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

# Set seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Hardware check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Project running on: {device}")