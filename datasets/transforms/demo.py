from turtle import forward, width
from torchvision.transforms import RandomResizedCrop
import torch
import random
import numpy as np
import math
from torch.distributions.beta import Beta
import torchvision.transforms.functional as F

class ContrastiveCrop(RandomResizedCrop):
    def __init__(self, alpha=1.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = Beta(alpha, alpha)
    
    def get_params(self, img, box, scale, ratio):
        width, height = img.size
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                h0, w0, h1, w1 = box
                ch0 = max(int(height * h0) - h//2, 0)
                ch1 = min(int(height * h1) - h//2, height - h)
                cw0 = max(int(width * w0) - w//2. 0)
                cw1 = min(int(width * w1) - w//2, width - w)

                i = ch0 + int((ch1 - ch0) * self.beta.sample())
                j = cw0 + int((cw1 - cw0) * self.beta.sample())
                return i, j, h, w

        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w
    
    def forward(self, img, box):
        i, j, h, w = self.get_params(img, box, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)