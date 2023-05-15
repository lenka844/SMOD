from statistics import mean
from matplotlib.transforms import TransformNode
from sklearn.preprocessing import scale
from torchvision import transforms
from .ContrastiveCrop import ContrastiveCrop
from .misc import MultiViewTransform, CCompose
import math
import torch
import cv2
import numpy as np
from PIL import ImageFilter
import random
import albumentations as A

np.seterr(divide='ignore',invalid='ignore')
from torch import Tensor

def underwater_train_full_rcrop(mean=None, std=None):
    trans_list = [
        transforms.RandomResizedCrop(size=64, scale=(0.2,1.0)),
        # transforms.RandomResizedCrop(size=128, scale=(0.2,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        RecoverDefog(p=0.8),
        RecoverMoblur(p=0.8),
        RecoverSharpen(p=0.8),
        RecoverHE(p=0.5),
        RecoverGC(p=0.8),
        RecoverCLAHE(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    transform = transforms.Compose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def underwater_train_full_ccrop(alpha=0.6, mean=None, std=None):
    trans_list = [
        ContrastiveCrop(alpha=alpha, size=64, scale=(0.2, 1.0)),
        # ContrastiveCrop(alpha=alpha, size=128, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        RecoverHE(p=0.5),
        RecoverGC(p=0.8),
        RecoverCLAHE(p=0.5),
        RecoverDefog(p=0.8),
        RecoverMoblur(p=0.8),
        RecoverSharpen(p=0.8),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    transform = CCompose(trans_list)
    transform = MultiViewTransform(transform, num_views=2)
    return transform

def underwater_full_linear(mean, std):
    trans = transforms.Compose([
        # transforms.RandomResizdCrop(size=32),
        transforms.RandomResizdCrop(size=64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return trans

def underwater_full_test(mean, std):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform



# augmentation strategies

def RecoverCLAHE_fir(sceneRadiance):
    sceneRadiance = np.array(sceneRadiance)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4, 4))
    for i in range(3):
        # sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))


    return sceneRadiance

def RecoverCLAHE_sec(img: Tensor) -> Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverCLAHE_fir(img)
    return RecoverCLAHE_fir(img)


class RecoverCLAHE(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverCLAHE_sec(img)
        return img
    
    def __repr__(self) -> str:
        return self.__calss__.__name__+ '(p={})'.format(self.p)

def RecoverGC_fir(img):
    img = np.array(img)
    # print('66666666666666666', type(img))
    sceneRadiance = img/255.0
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 0.7)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance


def RecoverGC_sec(img: Tensor) -> Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverGC_fir(img)
    return RecoverGC_fir(img)


class RecoverGC(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverGC_sec(img)
        return img
    
    def __repr__(self) -> str:
        return self.__calss__.__name__+ '(p={})'.format(self.p)
def RecoverHE_sec(img: Tensor) -> Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverHE_fir(img)
    return RecoverHE_fir(img)
def RecoverHE_fir(sceneRadiance):
    sceneRadiance = np.array(sceneRadiance)
    # print('000000000000', type(sceneRadiance))
    # clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(2, 2))
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
        # sceneRadiance[:, :, i] = clahe.apply((sceneRadiance[:, :, i]))
    return sceneRadiance

class RecoverHE(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverHE_sec(img)
        return img
    
    def __repr__(self) -> str:
        return self.__calss__.__name__+ '(p={})'.format(self.p)




class GaussianBlurConv():
    '''
    高斯滤波
    依据图像金字塔和高斯可分离滤波器思路加速
    '''
    def FilterGaussian(self, img, sigma):
        '''
        高斯分离卷积，按照x轴y轴拆分运算，再合并，加速运算
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        # 获取滤波器尺寸且强制为奇数
        kernel_size = round(sigma * 3 * 2 +1) | 1   # 当图像类型为CV_8U的时候能量集中区域为3 * sigma,
        # 创建内核
        kernel = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma, ktype=cv2.CV_32F)
        # 初始化图像
        temp = np.zeros_like(img)
        # x轴滤波
        for j in range(temp.shape[0]):
            for i in range(temp.shape[1]):
                # 内层循环展开
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    source = math.floor(i+ kernel_size/2 -k)        # 把第i个坐标和kernel的中心对齐 -k是从右往左遍历kernel对应的图像，得到与kernel的第k个元素相乘的图像坐标
                    if source < 0:
                        source = source * -1            # 如果图像超出左边缘，就反向，对称填充
                    if source > img.shape[1]:
                        source = math.floor(2 * (img.shape[1] - 1) - source)   # 图像如果超出右边缘，就用左边从头数着补
                    v1 += kernel[k] * img[j, source, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * img[j, source, 1]
                    v3 += kernel[k] * img[j, source, 2]
                temp[j, i, 0] = v1
                if temp.shape[2] == 1: continue
                temp[j, i, 1] = v2
                temp[j, i, 2] = v3
        # 分离滤波，先在原图用x轴的滤波器滤波，得到temp图，再用y轴滤波在temp图上滤波，结果一致
        # y轴滤波
        for i in range(img.shape[1]):         # height
            for j in range(img.shape[0]):
                v1 = v2 = v3 = 0
                for k in range(kernel_size):
                    source = math.floor(j + kernel_size/2 - k)
                    if source < 0:
                        source = source * -1
                    if source > temp.shape[0]:
                        source = math.floor(2 * (img.shape[0] - 1) - source)   # 上下对称
                    v1 += kernel[k] * temp[source, i, 0]
                    if temp.shape[2] == 1: continue
                    v2 += kernel[k] * temp[source, i, 1]
                    v3 += kernel[k] * temp[source, i, 2]
                img[j, i, 0] = v1
                if img.shape[2] == 1: continue
                img[j, i, 1] = v2
                img[j, i, 2] = v3
        return img

    def FastFilter(self, img, sigma):
        '''
        快速滤波，按照图像金字塔，逐级降低图像分辨率，对应降低高斯核的sigma，
        当sigma转换成高斯核size小于10，再进行滤波，后逐级resize
        递归思路
        '''
        # reject unreasonable demands
        if sigma > 300:
            sigma = 300
        # 获取滤波尺寸，且强制为奇数
        kernel_size = round(sigma * 3 * 2 + 1) | 1  # 当图像类型为CV_8U的时候能量集中区域为3 * sigma,
        # 如果s*sigma小于一个像素，则直接退出
        if kernel_size < 3:
            return
        # 处理方式(1) 滤波  (2) 高斯光滑处理  (3) 递归处理滤波器大小
        if kernel_size < 10:
            # img = self.FilterGaussian(img, sigma)
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)   # 官方函数
            return img
        else:
            # 若降采样到最小，直接退出
            if img.shape[1] < 2 or img.shape[0] < 2:
                return img
            sub_img = np.zeros_like(img)        # 初始化降采样图像
            sub_img = cv2.pyrDown(img, sub_img)           # 使用gaussian滤波对输入图像向下采样，缩放二分之一，仅支持CV_GAUSSIAN_5x5
            sub_img = self.FastFilter(sub_img, sigma/2.0)
            img = cv2.resize(sub_img, (img.shape[1], img.shape[0]))              # resize到原图大小
            return img

    def __call__(self, x, sigma):
        x = self.FastFilter(x, sigma)
        return x

class Retinex(object):
    """
    SSR: baseline
    MSR: keep the high fidelity and the dynamic range as well as compressing img
    MSRCR_GIMP:
      Adapt the dynamics of the colors according to the statistics of the first and second order.
      The use of the variance makes it possible to control the degree of saturation of the colors.
    """
    def __init__(self, model='MSR', sigma=[30, 150, 300], restore_factor=2.0, color_gain=10.0, gain=270.0, offset=128.0):
        self.model_list = ['SSR','MSR']
        if model in self.model_list:
            self.model = model
        else:
            raise ValueError
        self.sigma = sigma        # 高斯核的方差
        # 颜色恢复
        self.restore_factor = restore_factor     # 控制颜色修复的非线性
        self.color_gain = color_gain             # 控制颜色修复增益
        # 图像恢复
        self.gain = gain           # 图像像素值改变范围的增益
        self.offset = offset       # 图像像素值改变范围的偏移量
        self.gaussian_conv = GaussianBlurConv()   # 实例化高斯算子

    def _SSR(self, img, sigma):
        filter_img = self.gaussian_conv(img, sigma)    # [h,w,c]
        retinex = np.log10(img) - np.log10(filter_img)
        return retinex

    def _MSR(self, img, simga):
        retinex = np.zeros_like(img)
        for sig in simga:
            retinex += self._SSR(img, sig)
        retinex = retinex / float(len(self.sigma))
        return retinex

    def _colorRestoration(self, img, retinex):
        img_sum = np.sum(img, axis=2, keepdims=True)  # 在通道层面求和
        # 颜色恢复
        # 权重矩阵归一化 并求对数，得到颜色增益
        color_restoration = np.log10((img * self.restore_factor / img_sum) * 1.0 + 1.0)
        # 将Retinex做差后的图像，按照权重和颜色增益重新组合
        img_merge = retinex * color_restoration * self.color_gain
        # 恢复图像
        img_restore = img_merge * self.gain + self.offset
        return img_restore

    def _simplestColorBalance(self, img, low_clip, high_clip):
        total = img.shape[0] * img.shape[1]
        for i in range(img.shape[2]):
            unique, counts = np.unique(img[:, :, i], return_counts=True)  # 返回新列表元素在旧列表中的位置，并以列表形式储存在s中
            current = 0
            for u, c in zip(unique, counts):
                if float(current) / total < low_clip:
                    low_val = u
                if float(current) / total < high_clip:
                    high_val = u
                current += c
            img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
        return img

    def _MSRCR_GIMP(self, img):
        self.img = np.float32(img) + 1.0
        if self.model == 'SSR':
            self.retinex = self._SSR(self.img, self.sigma)
        elif self.model == 'MSR':
            self.retinex = self._MSR(self.img, self.sigma)
        # 颜色恢复 图像恢复
        self.img_restore = self._colorRestoration(self.img, self.retinex)

        return self.img_restore

    def __call__(self, img):
        return self._MSRCR_GIMP(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '{},sigma={},dynamic={}'.format(self.model, self.sigma)
        return repr_str

def RecoverRetinex_sec(img:Tensor)->Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverRetinex_fir(img)
    return RecoverRetinex_fir(img)

def RecoverRetinex_fir(img):
    retinex = Retinex()
    img = np.array(img)
    img = retinex(img)
    return img
class RecoverRetinex(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverRetinex_sec(img)
        return img
    
    def __repr__(self)->str:
        return self.__class__.name__+'(p={})'.format(self.p)

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    '''if r <= 0:
        return src
    h, w = src.shape[:2]
    I = src
    res = np.minimum(I  , I[[0]+range(h-1)  , :])
    res = np.minimum(res, I[range(1,h)+[h-1], :])
    I = res
    res = np.minimum(I  , I[:, [0]+range(w-1)])
    res = np.minimum(res, I[:, range(1,w)+[w-1]])
    return zmMinFilterGray(res, r-1)'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))  # 使用opencv的erode函数更高效
 
 
def guidedfilter(I, p, r, eps):
    '''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p
 
    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I
 
    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I
 
    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b
 
 
def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = （1-t）A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
 
    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制
 
    return V1, A
 
 
def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y

def RecoverDefog_sec(img:Tensor)->Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverDefog_fir(img)
    return RecoverDefog_fir(img)

def RecoverDefog_fir(img):
    img = np.array(img)
    img = deHaze(img / 255.0) * 255
    img = np.uint8(img)
    return img

class RecoverDefog(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverDefog_sec(img)
        return img
    
    def __repr__(self)->str:
        return self.__class__.name__+'(p={})'.format(self.p)

def RecoverMoblur_sec(img:Tensor)->Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverMoblur_fir(img)
    return RecoverMoblur_fir(img)

def RecoverMoblur_fir(img):
    img = np.array(img)
    size = 30
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    img = cv2.filter2D(img, -1, kernel_motion_blur)
    return img

class RecoverMoblur(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1).item() < self.p:
            return RecoverMoblur_sec(img)
        return img
    
    def __repr__(self)->str:
        return self.__class__.name__+'(p={})'.format(self.p)

def RecoverSharpen_fir(img):
    img = np.array(img)
    sharpen = A.Sharpen(alpha=(0, 1.0), lightness=(1.2, 11.5))
    img = sharpen(image=img)
    return img['image']

def RecoverSharpen_sec(img:Tensor)->Tensor:
    if not isinstance(img, torch.Tensor):
        return RecoverSharpen_fir(img)
    return RecoverSharpen_fir(img)

class RecoverSharpen(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1) < self.p:
            return RecoverSharpen_sec(img)
        return img
    
    def __repr__(self)->str:
        return self.__calss__.__name__+ '(p={})'.format(self.p)
