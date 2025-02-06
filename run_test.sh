#!/bin/bash

python main.py main data-set --name CelebA classifier --path checkpoints/classifiers/CelebA_CNN_9.pth generative-model --g_type Flow adv-attack --image_path images/resize_000082.jpg --target_class 1 --lr 5e-3 --num_steps 1000 --save_at 0.99