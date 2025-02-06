import os
import time
import torch
import datetime
import numpy as np

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import transforms

import cv2
import PIL
from .unet import unet
from PIL import Image


from .utils import *

def transformer(resize, totensor, normalize, centercrop, imsize):
    options = []
    if centercrop:
        options.append(transforms.CenterCrop(160))
    if resize:
        options.append(transforms.Resize((imsize,imsize), interpolation=PIL.Image.NEAREST))
    if totensor:
        options.append(transforms.ToTensor())
    if normalize:
        options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(options)
    
    return transform

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    f = dir.split('/')[-1].split('_')[-1]
    print (dir, len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]))
    for i in range(len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])):
        img = str(i) + '.jpg'
        path = os.path.join(dir, img)
        images.append(path)
   
    return images

class Tester(object):
    def __init__(self, config):
        # exact model and loss
        self.model = config.model

        # Model hyper-parameters
        # self.imsize = config.imsize
        self.parallel = config.parallel

        # self.total_step = config.total_step
        # self.batch_size = config.batch_size
        # self.num_workers = config.num_workers
        # self.g_lr = config.g_lr
        # self.lr_decay = config.lr_decay
        # self.beta1 = config.beta1
        # self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.img_path = config.img_path
        # self.label_path = config.label_path 
        self.log_path = config.log_path
        # self.model_save_path = config.model_save_path
        self.sample_path = config.sample_path
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        # self.model_save_step = config.model_save_step
        self.version = config.version

        # Path
        self.log_path = os.path.join(config.log_path, self.version)
        self.sample_path = os.path.join(config.sample_path, self.version)
        # self.model_save_path = os.path.join(config.model_save_path, self.version)
        self.test_label_path = config.test_label_path
        self.test_color_label_path = config.test_color_label_path
        self.test_image_path = config.test_image_path

        # Test size and model
        self.test_size = config.test_size
        self.model_name = config.model_name

        self.build_model()

    def test(self, image: torch.Tensor):
        # 2) [C, H, W] 形式の場合、バッチ次元 [1, C, H, W] を追加
        if image.ndim == 3:  # 画像が [C, H, W] の場合
            img_tensor = image.unsqueeze(0)
        elif image.ndim == 4:  # 既に [N, C, H, W] の場合はそのまま
            img_tensor = image
        else:
            raise ValueError(f"Unexpected input shape: {image.shape}")

        # 3) GPU に送る
        img_tensor = img_tensor.cuda()

        # 4) 推論
        self.G.eval()
        labels_predict = self.G(img_tensor)

        # 5) セグメンテーション結果（plain, color）の生成
        labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        labels_predict_color = generate_label(labels_predict, self.imsize)

        # 6) 保存
        cv2.imwrite(os.path.join(self.test_label_path, 'predict.png'),
                labels_predict_plain[0])
        save_image(labels_predict_color[0],
               os.path.join(self.test_color_label_path, 'predict_color.png'))

        print("Single-image test done.")

    def build_model(self):
        self.G = unet().cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        # print networks
        print(self.G)
