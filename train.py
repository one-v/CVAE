import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import defaultdict
from utils import create_dataset

from models import VAE

# 解决 OpenMP 库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    # 1、创建数据集，获取数据集的输入特征维度和输出标签维度
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    data_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    def loss_fn(recon_x, x, mean, log_var):
        # 使用 Smooth L1  + KLD 作为最终损失函数
        SmoothL1 = torch.nn.functional.smooth_l1_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (SmoothL1 + KLD) / x.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    # 2、选择 Adam 优化器
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        tracker_epoch = defaultdict(lambda: defaultdict(dict))

        for iteration, (x, y) in enumerate(data_loader):

            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # 用于潜在空间可视化，可以先注释掉
            # for i, yi in enumerate(y):
            #     id = len(tracker_epoch)
            #     tracker_epoch[id]['x'] = z[i, 0].item()
            #     tracker_epoch[id]['y'] = z[i, 1].item()
            #     tracker_epoch[id]['label'] = yi[0].item()

            loss = loss_fn(recon_x, x, mean, log_var)

            # 梯度清零、反向传播、参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

    # 6 走到这里，说明多轮训练结束，保存模型（参数）
    # 参1：模型对象的参数（权重矩阵，偏置矩阵） 参2：模型保存的文件名
    print(f"\n保存的模型参数为：{vae.state_dict()}\n")
    torch.save(vae.state_dict(), f"./model/CVAE.pth")  # 后缀名用:pth,pkl,pickle即可


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    # 输入：6个光学特性
    parser.add_argument("--encoder_layer_sizes", type=list, default=[6, 128, 64])  # 6+6=12
    # 条件：6个光学特性（作为条件） 输出：6个结构参数
    parser.add_argument("--decoder_layer_sizes", type=list, default=[64, 128, 6])   # 输出6个参数
    parser.add_argument("--latent_size", type=int, default=4)  # 增加潜在空间维度
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true', default=True)

    args = parser.parse_args() 

    main(args)
