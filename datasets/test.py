import torch
from PIL import Image
from .dataset import UnderWater
from pylab import *
import cv2
import os
import random
class test_boxes(UnderWater):
    def __init__(self, root, train, transform_rcrop, transform_ccrop, init_box=[0., 0., 1., 1.], **kwargs):
        super().__init__(train=train, root=root, **kwargs)
        self.transform_rcrop = transform_rcrop
        self.transform_ccrop = transform_ccrop
        self.boxes = torch.tensor(init_box).repeat(self.__len__(),1)
        self.use_box = True

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        img = self.data[index]
        img = array(Image.open(img))
        target = self.targets[index]
        img = Image.fromarray(img)

        if self.use_box:
            box = self.boxes[index].float().tolist()
            img = self.transform_ccrop([img, box])
            for image in img:
                image = torch.transpose(image, 0, 2)
                image = image.numpy()
                image = image * 255
                save_root = '/home/ljh/self-detection/ContrastiveCrop/Ccrop_demo'
                img_name = self.data[index].split('/')[-1].split('.')[0]
                # print('-------',img_name)
                save_path = os.path.join(save_root, img_name + '.jpg')
                image = cv2.resize(image, (224,224))
                cv2.imwrite(save_path, image)
        else:
            img = self.transform_rcrop(img)
            for image in img:
                image = torch.transpose(image, 0, 2)
                image = image.numpy()
                image = image * 255
                save_root = '/home/ljh/self-detection/ContrastiveCrop/Rcrop_demo'
                img_name = self.data[index].split('/')[-1].split('.')[0]
                save_path = os.path.join(save_root, img_name+'.jpg')
                image = cv2.resize(image, (224,224))
                # print('----the image size is {}'.format(image.shape))
                cv2.imwrite(save_path, image)
                # image = Image.fromarray(np.uint8(image))
                # print('====the type of pic is {}'.format(type(image)))
                # r, g, b = image.getextrema()
                # if r[1] == 0 and g[1] == 0 and b[1] == 0:
                #     print('---There is a balck picture!')
                # print('----------', (image.shape))
                # print('---------', image.getextrema())
        # for image in img:
        #     image = torch.transpose(image, 0, 2)
        #     image = image.numpy()
        #     # image = image.transpose(image, (1, 2, 0))
        #     print('------the type of img is {}'.format(type(image)))
        #     # print(image.shape)
        #     # print('----------', type(image))
        #     save_root = '/home/ljh/self-detection/ContrastiveCrop/Ccrop_demo'
        #     save_path = os.path.join(save_root, str(random.random())+'.jpg')
        #     cv2.imwrite(save_path, image)
        
        return img, target
