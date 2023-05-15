import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
from np import *
from pylab import *

class UnderWater(Dataset):
    def __init__(self, root,  train=True, transform=None):
        super().__init__()
        self.root = root
        self.img_path = os.listdir(self.root)
        self.length = 0
        self.transform = transform
        self.data: Any = []
        self.targets = []
        if train:
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        for img in os.listdir(self.img_path):
            self.length += 1
            self.data.append(os.path.join(self.img_path, img))
            self.targets.append('cat')
        print('dataset_size:', self.length)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image = self.data[idx]
        image = np.array(Image.open(image))
        image = Image.fromarray(image).resize((64, 64))
        # print('==========',image.size)
        label = 0 if self.data[idx].split('.')[0] =='cat' else 1
        if self.transform:
            image = self.transform(image)
        target = torch.from_numpy(np.array([label]))
        return image, target