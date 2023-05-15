from statistics import mean
from matplotlib.transforms import TransformNode
from sklearn.preprocessing import scale
from torchvision import transforms
from .ContrastiveCrop import ContrastiveCrop
from .misc import MultiViewTransform, CCompose

def underwater_train_rcrop(mean=None, std=None):
    trans_list = [
        transforms.RandomResizedCrop(size=64, scale=(0.2,1.0)),
        # transforms.RandomResizedCrop(size=128, scale=(0.2,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform = transforms.Compose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def underwater_train_ccrop(alpha=0.6, mean=None, std=None):
    trans_list = [
        ContrastiveCrop(alpha=alpha, size=64, scale=(0.2, 1.0)),
        # ContrastiveCrop(alpha=alpha, size=128, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform = CCompose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def underwater_linear(mean, std):
    trans = transforms.Compose([
        # transforms.RandomResizdCrop(size=32),
        transforms.RandomResizdCrop(size=128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans

def underwater_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform