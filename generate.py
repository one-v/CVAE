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


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # 当前不做数据归一化处理
    # if not os.path.exists('./scalers.pkl'):
    #     print("StandardScaler文件不存在，请先运行train.py生成scalers.pkl文件")
    #     return
    # with open('./scalers.pkl', 'rb') as f:
    #     scalers = pickle.load(f)
    # x_scaler = scalers['x_scaler']
    # y_scaler = scalers['y_scaler']
    # print("已加载StandardScaler对象")

    # 生成n组解
    generated_solutions = []

    with torch.no_grad():
        for i in range(args.num_samples):
            # 生成随机潜在变量z
            z = torch.randn(1, args.latent_size).to(device)

            # 如果是条件VAE，需要提供条件c
            if args.conditional:
                # 这里可以根据需要修改条件c的生成方式
                # 示例：生成条件c: 6个光学特性
                nums_1 = [4.79721214, 5.885984774, 1.184874056, 0.491146023, 0.293180056, 1.384228451]
                nums_2 = [3.500724077, 4.191257477, 0.796704531, 0.35748297, -0.005191585, 0.483815134]
                nums_3 = [8.567868233, 12.92377758, 5.244325638, 3.140165329, 1.899796963, 0.822457731]

                # 转换为PyTorch张量（CPU设备，float32类型）
                # 不做反归一化处理
                # c_normalized = y_scaler.inverse_transform(np.array(nums, dtype=np.float32).reshape(1, -1))
                c = torch.tensor(nums_1, dtype=torch.float32).unsqueeze(0).to(device)
                generate_x = vae.inference(z, c)
                generate_x[:,0] = generate_x[:,0]/100   # 第0列：n1还原
                generate_x[:,1] = generate_x[:,1]/100   # 第1列：n2还原
                # recon_x_original = x_scaler.inverse_transform(recon_x.cpu().numpy())
                generated_solutions.append({
                    '光学特性参数': c[0].tolist(),
                    '生成的光纤结构参数': generate_x[0].tolist()
                })
            else:
                generate_x = vae.inference(z)
                # recon_x_original = x_scaler.inverse_transform(recon_x.cpu().numpy())
                generated_solutions.append({
                    '生成的光纤结构参数': generate_x[0].tolist()
                })

            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1} 组解")


    # 打印生成的解作为示例
    print(f"生成的{args.num_samples}组解:")
    for i, solution in enumerate(generated_solutions[:args.num_samples]):
        if '光学特性参数' in solution:
            print(f"第{i + 1}组:")
            print(f"条件(光学特性参数):\t{solution['光学特性参数']}")
            print(f"解(生成的光纤结构参数):\t{solution['生成的光纤结构参数']}\n")
        else:
            print(f"第{i + 1}组:\t{solution['生成的光纤结构参数']}") 

    # 保存生成的解到CSV文件
    output_df = pd.DataFrame(generated_solutions)
    output_path = f"./generate_solution/CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}_{args.num_samples}.csv"
    output_df.to_csv(output_path, index=False)
    print(f"生成的解已保存到: {output_path}")


if __name__ == '__main__':
    # 获取模型列表
    train_list = model_list()
    # 设置生成解的个数
    num_samples = 50
    for train_dict in train_list:
        # 判断是否训练该模型
        if not train_dict["if_train"]:
            continue
        # 解析命令行参数
        args = get_args(train_dict,num_samples)
        print(
            f"当前模型：CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}")
        # 从所有模型生成解
        generate(args)
