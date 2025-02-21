import os
import time
import torch
import datetime
import numpy as np
import imageio

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
        self.imsize = config.imsize
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

        # GIF 画像を保存するディレクトリ
        self.gif_save_path = "./results/gif"
        os.makedirs(self.gif_save_path, exist_ok=True)

        # セグメンテーション画像を保存するリスト
        self.segmented_images = []
        
        self.build_model()

    def test(self, image: torch.Tensor, step: int = None):
        # [C, H, W] の場合、バッチ次元を追加
        img_tensor = image.unsqueeze(0) if image.ndim == 3 else image
        img_tensor = img_tensor.cuda()

        # 推論
        self.G.eval()
        labels_predict = self.G(img_tensor)

        # セグメンテーション結果の生成
        labels_predict_plain = generate_label_plain(labels_predict, self.imsize)
        labels_predict_color = generate_label(labels_predict, self.imsize)

        # 🔹 保存時のファイル名変更
        step_suffix = f"_step_{step}" if step is not None else ""
    
        cv2.imwrite(os.path.join(self.test_label_path, f'predict{step_suffix}.png'),
                labels_predict_plain[0])
        save_image(labels_predict_color[0],
               os.path.join(self.test_color_label_path, f'predict_color{step_suffix}.png'))

        # 🔹 GIF 用に画像を保存
        self.segmented_images.append(labels_predict_color[0].cpu().permute(1, 2, 0).numpy())

        print(f"Single-image test done. Saved as predict{step_suffix}.png")

    def build_model(self):
        """ モデルを構築し、学習済み重みをロードする """

        # 🔹 モデルを初期化
        self.G = unet().cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)

        print("== モデル構造 ==")
        print(self.G)  # モデル構造を出力

        # 🔹 学習済みモデルをロード
        model_path = "segmentation/models/parsenet/model.pth"

        print(f"Loading model weights from: {model_path}")

        try:
            checkpoint = torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

            # 🔹 チェックポイントの内容を確認
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.G.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.G.load_state_dict(checkpoint, strict=False)

            print("✅ モデルのロード成功")
        except Exception as e:
            print(f"❌ モデルのロードに失敗しました: {e}")
    
        # 🔹 学習済みの重みが適用されているか確認
        for name, param in self.G.named_parameters():
            print(f"{name}: {param.mean().item()}")  # 平均値を表示
            break  # 1つのパラメータのみ確認すればOK

    def get_segmentation_prob(self, image: torch.Tensor):
        """
        画像に対してセグメンテーションを適用し、各クラスの確率分布を返す
        """
        self.G.eval()
        # with torch.no_grad():
        logits = self.G(image.cuda())  # [N, C, H, W]
        probs = torch.nn.functional.softmax(logits, dim=1)  # 確率分布に変換
        return probs

    def save_gif(self):
        """保存したセグメンテーション画像から GIF を作成"""
        gif_path = os.path.join(self.gif_save_path, "segmentation_animation.gif")

        # 🔹 画像を uint8 に変換
        segmented_images_uint8 = [(img * 255).astype(np.uint8) for img in self.segmented_images]

        imageio.mimsave(gif_path, segmented_images_uint8, duration=0.2)
        print(f"GIF saved at: {gif_path}")