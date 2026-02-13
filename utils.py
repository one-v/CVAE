# 导包
import argparse
import torch  # 封装了对张量的各种操作
import numpy as np  # 对矩阵进行操作
import pandas as pd  # 对数据集进行操作
from torch.utils.data import TensorDataset  # 数据集对象 数据→Tensor→数据集→数据加载器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
from sklearn.preprocessing import StandardScaler


# 创建数据集
def create_dataset():
    # 1、加载csv文件数据集文件
    data = pd.read_excel('./dataset/Δneff_comsol+网络.xlsx')
    # 只使用前11545行数据，因为前面的数据是真的仿真数据，11545行后的数据是神经网络生成的数据
    data = data.iloc[:11545, :-1]

    # 2、获取x特征列和y标签列同时把特征列转成浮点型，一定要转成浮点型，因为在神经网络中，计算都是浮点型   
    # x：6个光学特性（作为输入）  y：6个结构参数（作为输出）
    x = data.iloc[:, :6].astype(np.float32)  # 光学结构参数即作为输入，也作为输出
    # 对x的前两列乘以100倍
    x.iloc[:, :2] = x.iloc[:, :2] * 100
    y = data.iloc[:, 6:].astype(np.float32)  # 光学特性参数作为条件输入

    # 3、切分训练集和测试集
    # 参1：特征 参2：标签 参3：测试集所占比例 参4：随机种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)
    x = x*10000
    y = y*10000

    # 先不对x,y做标准化处理
    # x_scaler = StandardScaler()
    # x_train = x_scaler.fit_transform(x_train)
    # x_test = x_scaler.transform(x_test)
    # # 对y做标准化处理
    # y_scaler = StandardScaler()
    # y_train = y_scaler.fit_transform(y_train)
    # y_test = y_scaler.transform(y_test)

    # 4、把数据集封装成 张量数据集 思路：数据 → 张量Tensor → 数据集TensorDataSet → 数据加载器DataLoader
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test.values), torch.from_numpy(y_test.values))

    # 只有做了标准化处理，才需要保存StandardScaler对象
    # with open('./scalers.pkl', 'wb') as f:
    #     pickle.dump({'x_scaler': x_scaler, 'y_scaler': y_scaler}, f)
    # print("StandardScaler对象已保存到 ./scalers.pkl")

    return train_dataset, test_dataset, x_train.shape[1], y_train.shape[1]


# 构建命令行参数解析器，用于后续模型传参
def get_args(train_dict, num_samples=10):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=train_dict["epochs"])
    parser.add_argument("--batch_size", type=int, default=train_dict["batch_size"])
    parser.add_argument("--learning_rate", type=float, default=train_dict["learning_rate"])
    # 输入：6个光学特性
    parser.add_argument("--encoder_layer_sizes", type=list, default=train_dict["encoder_layer_sizes"])  # 6+6=12
    # 条件：6个光学特性（作为条件） 输出：6个结构参数
    parser.add_argument("--decoder_layer_sizes", type=list, default=train_dict["decoder_layer_sizes"])  # 输出6个参数
    parser.add_argument("--latent_size", type=int, default=train_dict["latent_size"])  # 增加潜在空间维度
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", action='store_true', default=True)
    # 在生成解时，需要指定生成的解的个数
    parser.add_argument("--num_samples", type=int, default=num_samples)

    return parser.parse_args()


# 添加模型构建器，用于快速构建模型结构
def model_list():
    model_list = []
    # 添加模型开关if_train，用于控制是否训练该模型
    # 模型大概在10000轮之后已经过拟合，所以轮次暂时设置在5000左右即可
    for latent_size in [3, 9, 14, 16, 24]:
        for learning_rate in [0.01, 0.005, 0.001]:
            for batch_size in [16, 32]:
                for epochs in [10000, 20000, 30000]:
                    for encoder_layer_sizes in [[6, 256, 128], [6, 128, 64], [6, 256, 128, 64],
                                                [6, 128, 64, 32], [6, 256, 128, 64, 32]]:
                        model_list.append({
                            "encoder_layer_sizes": encoder_layer_sizes,
                            "decoder_layer_sizes": encoder_layer_sizes[::-1],
                            "latent_size": latent_size,
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "if_train": True
                        })
    # count = 0
    # for model in model_list:
    #     print(model)
    #     count += 1
    # print("模型总数：", count)

    # model_list = [
    #     # ------------- 原有经典推荐 -------------
    #     {"encoder_layer_sizes": [6, 256, 160, 80], "decoder_layer_sizes": [80, 160, 256, 6], "latent_size": 14,
    #      "learning_rate": 0.0008, "epochs": 10000, "batch_size": 16, "if_train": True},
    #     {"encoder_layer_sizes": [6, 128, 64], "decoder_layer_sizes": [64, 128, 6], "latent_size": 10,
    #      "learning_rate": 0.001, "epochs": 10000, "batch_size": 32, "if_train": True},
    #     {"encoder_layer_sizes": [6, 256, 128, 64], "decoder_layer_sizes": [64, 128, 256, 6], "latent_size": 12,
    #      "learning_rate": 0.001, "epochs": 10000, "batch_size": 16, "if_train": True},
    #     # ------------- 精度优先 -------------
    #     {"encoder_layer_sizes": [6, 128, 64, 32], "decoder_layer_sizes": [32, 64, 128, 6], "latent_size": 10,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     {"encoder_layer_sizes": [6, 256, 160, 64], "decoder_layer_sizes": [64, 160, 256, 6], "latent_size": 12,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     # ------------- 精度+多样性均衡（核心） -------------
    #     {"encoder_layer_sizes": [6, 256, 128, 80], "decoder_layer_sizes": [80, 128, 256, 6], "latent_size": 12,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     {"encoder_layer_sizes": [6, 256, 140, 70], "decoder_layer_sizes": [70, 140, 256, 6], "latent_size": 14,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     # ------------- 多样性优先（解决解单一） -------------
    #     {"encoder_layer_sizes": [6, 320, 180, 90], "decoder_layer_sizes": [90, 180, 320, 6], "latent_size": 14,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     {"encoder_layer_sizes": [6, 320, 200, 100], "decoder_layer_sizes": [100, 200, 320, 6], "latent_size": 16,
    #      "learning_rate": 0.005, "epochs": 8000, "batch_size": 32, "if_train": True},
    #     # 自定义结构
    #     {"encoder_layer_sizes": [6, 128, 64], "decoder_layer_sizes": [64, 128, 6], "latent_size": 12,
    #      "learning_rate": 0.01, "epochs": 5000, "batch_size": 32, "if_train": True},

    #     {"encoder_layer_sizes": [6, 256, 128], "decoder_layer_sizes": [128, 256, 6], "latent_size": 6,
    #      "learning_rate": 0.001, "epochs": 5000, "batch_size": 64, "if_train": True},
    #     {"encoder_layer_sizes": [6, 256, 128, 64], "decoder_layer_sizes": [64, 128, 256, 6], "latent_size": 6,
    #      "learning_rate": 0.001, "epochs": 5000, "batch_size": 64, "if_train": True},
    #     {"encoder_layer_sizes": [6, 256, 64], "decoder_layer_sizes": [64, 256, 6], "latent_size": 6,
    #      "learning_rate": 0.001, "epochs": 5000, "batch_size": 64, "if_train": True}
    # ]
    return model_list


if __name__ == '__main__':
    # create_dataset()
    model_list()
