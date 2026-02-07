import os
import torch
import argparse
import numpy as np
import pandas as pd
import pickle
from utils import model_list, get_args
from models import VAE

# 解决 OpenMP 库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 用来生成解的函数
def generate(args,c):

    # 创建VAE模型实例
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    # 加载训练好的模型参数
    model_path = f"./model/CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}.pth"

    # 确保model目录与模型文件存在
    if not os.path.exists('./model'):
        raise FileNotFoundError("模型目录'./model'不存在，请先运行train.py训练模型")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先运行train.py训练模型")
    # 加载模型参数
    vae.load_state_dict(torch.load(model_path))
    # 切换模型状态
    vae.eval()

    # 生成n组解
    generated_solutions = []

    with torch.no_grad():
        for i in range(args.num_samples):
            # 生成随机潜在变量z
            z = torch.randn(1, args.latent_size).to(device)

            # 如果是条件VAE，需要提供条件c
            if args.conditional:
                generate_x = vae.inference(z, c)
                
                generate_x[:,0] = generate_x[:,0]/100   # 第0列：n1还原
                generate_x[:,1] = generate_x[:,1]/100   # 第1列：n2还原
                generated_solutions.append(generate_x[0].tolist())
            else:
                generate_x = vae.inference(z)
                generated_solutions.append(generate_x[0].tolist())

            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1} 组解")

    return generated_solutions


if __name__ == '__main__':
    # 获取模型列表
    train_list = model_list()
    # 设置生成解的个数
    num_samples = 10
    # 设置要生成解的样本索引
    index = 5
    data = pd.read_excel('./dataset/Δneff_comsol+网络.xlsx')
    data = data.iloc[index,6:-1].astype(np.float32)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    for train_dict in train_list:
        # 判断是否训练该模型
        if not train_dict["if_train"]:
            continue
        # 解析命令行参数
        args = get_args(train_dict,num_samples)
        print(
            f"当前模型：CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}")
        # 从所有模型生成解
        generate(args,data)
