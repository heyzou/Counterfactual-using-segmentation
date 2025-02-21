from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
from typing import TypeVar, Dict
import numpy as np
import os

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from counterfactuals.utils import make_dir, get_transforms, torch_to_image, expl_to_image
from counterfactuals.plot import plot_grid_part
from counterfactuals.generative_models.base import GenerativeModel
from counterfactuals.classifiers.base import NeuralNet

# add segmentation dir
from segmentation.tester import Tester
from segmentation.parameter import *

Tensor = TypeVar('torch.tensor')

matplotlib.use('Agg')

def adv_attack(g_model: GenerativeModel,
               classifier: NeuralNet,
               device: str,
               attack_style: str,
               data_info: Dict,
               num_steps: int,
               lr: float,
               save_at: float,
               target_class: int,
               image_path: str,
               result_dir: str,
               maximize: bool) -> None:
    """
    prepare adversarial attack in X or Z
    run attack
    save resulting adversarial example/counterfactual
    """
    # load image
    transforms = get_transforms(data_info["data_shape"])
    x = transforms(Image.open(image_path)).to(device)

    # define parameters that will be optimized
    params = []
    if attack_style == "z":
        
        # define z as params for derivative wrt to z
        z = g_model.encode(x)
        z = [z_i.detach() for z_i in z] if isinstance(z, list) else z.detach()
        x_org = x.detach().clone()
        z_org = [z_i.clone() for z_i in z] if isinstance(z, list) else z.clone()

        if type(z) == list:
            for z_part in z:
                z_part.requires_grad = True
                params.append(z_part)
        else:
            z.requires_grad = True
            params.append(z)
    else:
        # define x as params for derivative wrt x
        x_org = x.clone()
        x.requires_grad = True
        params.append(x)
        z = None

    print("\nRunning counterfactual search in Z ..." if attack_style == 'z'
          else "Running conventional adv attack in X ...")
    optimizer = torch.optim.Adam(params=params, lr=lr, weight_decay=0.0)

    # run the adversarial attack
    x_prime = run_adv_attack(x, z, optimizer, classifier, g_model, target_class,
                             attack_style, save_at, num_steps, maximize)

    if x_prime is None:
        print("Warning: Maximum number of iterations exceeded! Attack did not reach target value, returned None.")
        return

    # save results
    result_dir = make_dir(result_dir)
    image_name = image_path.split('/')[-1].split('.')[0]
    data_shape = data_info["data_shape"]
    cmap_img = "jet" if data_shape[0] == 3 else "gray"

    # calculate heatmap as difference dx between original and adversarial/counterfactual
    # TODO: dx to original or projection?
    heatmap = torch.abs(x_org - x_prime).sum(dim=0).sum(dim=0)

    all_images = [torch_to_image(x_org)]
    titles = ["$x$", "$x^\prime$", "$\delta x$"]
    cmaps = [cmap_img, cmap_img, 'coolwarm']
    if attack_style == 'z':
        all_images.append(torch_to_image(g_model.decode(z_org)))
        titles = ["$x$", "$g(g^{-1}(x))$", "$x^\prime$", "$\delta x$"]
        cmaps = [cmap_img, cmap_img, cmap_img, 'coolwarm']

    all_images.append(torch_to_image(x_prime))
    all_images.append(expl_to_image(heatmap))

    _ = plot_grid_part(all_images, titles=titles, images_per_row=4, cmap=cmaps)
    plt.subplots_adjust(wspace=0.03, hspace=0.01, left=0.03, right=0.97, bottom=0.01, top=0.95)

    g_model_name = f"_{type(g_model).__name__}" if g_model is not None else ""
    plt.savefig(result_dir + f'overview_{image_name}_{attack_style}{g_model_name}_save_at_{save_at}.png')

def run_adv_attack(x: Tensor,
                   z: Tensor,
                   optimizer: Optimizer,
                   classifier: NeuralNet,
                   g_model: GenerativeModel,
                   target_class: int,
                   attack_style: str,
                   save_at: float,
                   num_steps: int,
                   maximize: bool) -> Tensor:
    """
    run optimization process on x or z for num_steps iterations
    early stopping when save_at is reached
    if not return None
    """
    target = torch.LongTensor([target_class]).to(x.device)

    softmax = torch.nn.Softmax(dim=1)
    loss_fn = nn.CrossEntropyLoss()

    config = get_parameters()
    tester = Tester(config)
    
    # 画像更新前のセグメンテーション確率分布を保存
    if attack_style == "z":
        x_org = g_model.decode(z).detach()  # 更新前の画像
    else:
        x_org = x.clone().detach()

    # prob_before = tester.get_segmentation_prob(x_org)  # 確率分布を取得
    prob_before = tester.get_segmentation_prob(x).detach()  # 勾配を切る
    # tester.test(x, 0)

    # セグメンテーションロスの比重を変化させるリスト
    weight_factors = np.arange(5.0, 5.1, 0.1)

    # 結果を格納する辞書
    loss_results = {}

    for weight in weight_factors:
        print(f"\n🔹 Running attack with segmentation weight {weight:.1f}")

        steps = []
        total_loss_list = []
        cross_entropy_loss_list = []
        classifier_loss_list = []

        with tqdm(total=num_steps, desc=f"Weight {weight:.1f}") as progress_bar:
            for step in range(num_steps):
                optimizer.zero_grad()

                if attack_style == "z":
                    x = g_model.decode(z)
                    tester.test(x, step)

                # 画像更新後のセグメンテーション確率分布を取得
                prob_after = tester.get_segmentation_prob(x)

                # クロスエントロピーの計算
                cross_entropy = -(prob_after * torch.log(prob_before + 1e-8)).sum(dim=1).mean()

                # 画像の値を [0,1] に制限
                x = torch.clamp(x, min=0.0, max=1.0)

                if "UNet" in type(classifier).__name__:
                    _, regression = classifier(x)
                    loss = -regression if maximize else regression
                else:
                    prediction = classifier(x)
                    acc = softmax(prediction)[torch.arange(0, x.shape[0]), target]
                    loss = loss_fn(prediction, target)

                # セグメンテーションロスの影響を変更
                total_loss = 0.3 * loss + weight * cross_entropy

                # 🔹 ロスの履歴を記録
                steps.append(step)
                total_loss_list.append(total_loss.item())
                cross_entropy_loss_list.append(cross_entropy.item())
                classifier_loss_list.append(loss.item())

                progress_bar.set_postfix(
                    total_loss=total_loss.item(),
                    cross_entropy=cross_entropy.item(),
                    classifier_loss=loss.item(),
                    step=step + 1,
                    acc= acc.item()
                )
                progress_bar.update()

                # 早期停止
                if "UNet" not in type(classifier).__name__ and acc > save_at:
                    print(f"✅ Early stopping at step {step} for weight {weight:.1f}")
                    return x

                total_loss.backward()
                optimizer.step()

        # 🔹 各 weight の結果を保存
        loss_results[weight] = {
            "steps": np.array(steps),
            "total_loss": np.array(total_loss_list),
            "cross_entropy": np.array(cross_entropy_loss_list),
            "classifier_loss": np.array(classifier_loss_list)
        }

        print(f"Total Loss List (Weight {weight:.1f}): {total_loss_list}")
        # 🔹 各 weight ごとの詳細なグラフを保存
        plt.figure(figsize=(10, 5))
        plt.plot(steps, total_loss_list, label="Total Loss", linestyle='-', color='blue')
        plt.plot(steps, cross_entropy_loss_list, label="Segmentation Loss", linestyle='--', color='red')
        plt.plot(steps, classifier_loss_list, label="Counterfactual Loss", linestyle='-.', color='green')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Loss Trends (Weight = {weight:.1f})")
        os.makedirs("./results/", exist_ok=True)
        plt.savefig(f"./results/loss_plot_weight_{weight:.1f}.png")
        plt.close()

    # 🔹 すべての比重の影響を比較するグラフ
    plt.figure(figsize=(12, 6))
    for weight, data in loss_results.items():
        plt.plot(data["steps"], data["total_loss"], label=f"Weight {weight:.1f}")

    plt.xlabel("Steps")
    plt.ylabel("Total Loss")
    plt.legend()
    plt.title("Impact of Segmentation Loss Weight on Total Loss")
    plt.savefig("./results/loss_weight_impact.png")
    plt.close()

    return None