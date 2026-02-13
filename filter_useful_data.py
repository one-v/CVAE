import pandas as pd
from generate import generate
from utils import model_list, get_args
import os
import torch
import numpy as np
import pandas as pd
from utils import model_list, get_args
from models import VAE
import time

# 解决 OpenMP 库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def round_solution(solution):
    """
    对生成的解进行小数点位数约束
    参数说明：
    solution: 生成的光纤结构参数列表 [n1, n2, r8, r9, r10, wl]
    返回：格式化后的解
    """

    # n1 保留小数点后四位
    solution[0] = round(solution[0], 4)
    # n2 保留小数点后四位
    solution[1] = round(solution[1], 4)
    # r8 保留小数点后三位
    solution[2] = round(solution[2], 3)
    # r9 保留小数点后三位
    solution[3] = round(solution[3], 3)
    # r10 保留小数点后三位
    solution[4] = round(solution[4], 3)
    # wl 保留小数点后两位
    solution[5] = round(solution[5], 2)

    return solution


def filter_solution(solution):
    """
    过滤不在范围内的解
    参数说明：
    solution: 格式化后的光纤结构参数列表 [n1, n2, r8, r9, r10, wl]
    返回：布尔值，True表示符合范围要求，False表示不符合
    """
    n1, n2, r8, r9, r10, wl = round_solution(solution)

    # 检查各参数范围
    if not (0.018 < n1 < 0.09):
        return False
    if not (-0.05 < n2 < 0):
        return False
    if not (2.5 < r8 < 8.5):
        return False
    if not (5.5 < r9 < 14.5):
        return False
    if not (6.5 < r10 < 16.5):
        return False
    if not (0.5 <= r10 - r9):
        return False
    if not (1 <= r9 - r8):
        return False
    if not (1.5 < wl < 1.6):
        return False

    return True


def filter_useful_data(generated_solutions):
    useful_data = []
    for generate_x in generated_solutions:
        if filter_solution(generate_x):
            useful_data.append(generate_x)
    return useful_data


if __name__ == '__main__':
    # 获取模型列表
    train_list = model_list()
    # 设置生成解的个数
    num_samples = 20
    # 设置要生成解的样本索引
    index = 3000
    data = pd.read_excel('./dataset/Δneff_comsol+网络.xlsx')
    orign_data = torch.tensor(data.iloc[index, :6].astype(np.float32)).to(device)
    data = data.iloc[index, 6:-1].astype(np.float32)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)
    #输入获取生成解的个数
    num_samples = int(input("请输入要生成的解的个数："))
    print(f"当前的光学特性参数为：{data}\n原数据集中对应的光纤结构参数为：{orign_data}")
    for train_dict in train_list:
        # 判断是否训练该模型
        if not train_dict["if_train"]:
            continue
        # 解析命令行参数
        args = get_args(train_dict, num_samples)
        print(
            f"当前模型为：CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}")
        # 从所有模型生成解
        generated_solutions = generate(args, data)
        # 过滤有用解
        useful_data = filter_useful_data(generated_solutions)
        # 打印有用解
        print(f"初步过滤后得到的有用解数量: {len(useful_data)}")
        for i, solution in enumerate(useful_data):
            print(f"第{i + 1}组解:\t {solution}")
        time.sleep(1)
