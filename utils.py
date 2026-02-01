# 导包
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

    # 2、获取x特征列和y标签列
    # x：6个光学特性（作为输入）
    # y：6个结构参数（作为输出）
    x = data.iloc[:, 6:]  # 光学特性作为输入
    y = data.iloc[:, :6]  # 结构参数作为输出

    # 3、把特征列转成浮点型，一定要转成浮点型，因为在神经网络中，计算都是浮点型
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    # 4、切分训练集和测试集
    # 参1：特征 参2：标签 参3：测试集所占比例 参4：随机种子
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)

    # 优化：数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)
    # 因为CVAE模型的输入是x和y，所以y也需要标准化
    y_train = transfer.fit_transform(y_train)
    y_test = transfer.transform(y_test)

    # 5、把数据集封装成 张量数据集 思路：数据 → 张量Tensor → 数据集TensorDataSet → 数据加载器DataLoader
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    return train_dataset, test_dataset, x_train.shape[1], y_train.shape[1]
