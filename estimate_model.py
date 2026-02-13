"""
案例：
    ANN（人工神经网络）案例：手机价格分类案例
背景：
    基于手机的20列特征  →   预测手机的价格区间（4个区间），可以用机器学习来做，也可以用 深度学习来做

ANN案例的实现步骤：
    1、构建数据集
    2、搭建神经网络
    3、模型训练
    4、模型测试

优化思路：
    1、优化方法从SGD → Adam
    2、学习率从0.001 → 0.0001
    3、对数据进行标准化/批量归一化
    4、增加网络的深度
    5、增加每层的神经元数量
    6、调整训练的轮数
    7、......
"""


# 导包
import torch  # 封装了对张量的各种操作
import torch.nn as nn  # 封装了对神经网络的各种操作
import numpy as np  # 对矩阵进行操作
import pandas as pd  # 对数据集进行操作
from torch.utils.data import TensorDataset  # 数据集对象 数据→Tensor→数据集→数据加载器
from torch.utils.data import DataLoader  # 数据加载器
import torch.optim as optim  # 优化器
from sklearn.model_selection import train_test_split  # 训练集和测试集的划分
import matplotlib.pyplot as plot  # 绘图
import time  # 时间模块
from torchsummary import summary  # 模型结构可视化
from sklearn.preprocessing import StandardScaler

# todo 1:创建数据集
def create_dataset():
    # 1、加载csv文件数据集文件
    data = pd.read_excel('./dataset/Δneff_comsol+网络.xlsx')
    # 只使用前11545行数据，因为前面的数据是真的仿真数据，11545行后的数据是神经网络生成的数据
    data = data.iloc[:11545, :-1]
    # 输出数据集的形状和前五行数据
    # print(f"data.shape:{data.shape}\n")   # (2000, 21)
    # print(f"data.head():{data.head()}\n")

    # 2、获取x特征列和y标签列
    x = data.iloc[:, :6]
    y = data.iloc[:, 6:]
    # 对y中所有的列的值都乘以10000倍
    y = y * 10000
    # print(f"x.shape:{x.shape} x.dtypes:{x.dtypes}")
    # print(f"y.shape:{y.shape}")

    # 3、把特征列转成浮点型，一定要转成浮点型，因为在神经网络中，计算都是浮点型
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    # print(f"x.shape:{x.shape} x.dtypes:{x.dtypes}")

    # 4、切分训练集和测试集
    # 参1：特征 参2：标签 参3：测试集所占比例 参4：随机种子 参5：样本的分布（即：参考y的类别进行抽取数据）
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=32)
    x_train[['n1', 'n2']] = x_train[['n1', 'n2']] * 100
    x_test[['n1', 'n2']] = x_test[['n1', 'n2']] * 100

    # 优化：数据标准化
    # transfer = StandardScaler()
    # x_train = transfer.fit_transform(x_train)
    # x_test = transfer.transform(x_test)

    # 5、把数据集封装成 张量数据集 思路：数据 → 张量Tensor → 数据集TensorDataSet → 数据加载器DataLoader
    train_dataset = TensorDataset(torch.from_numpy(x_train.values), torch.from_numpy(y_train.values))
    test_dataset = TensorDataset(torch.from_numpy(x_test.values), torch.from_numpy(y_test.values))
    # print(f"train_dataset:{train_dataset}")
    # print(f"test_dataset:{test_dataset}")

    return train_dataset, test_dataset, x_train.shape[1], y_train.shape[1]


# todo 2:创建神经网络
class estimateNet(nn.Module):
    # 1、在init魔法方法中，初始化父类成员，及搭建神经网络
    def __init__(self, input_dim, output_dim):  # 输入：20 输出：4
        # 1.1 初始化父类成员
        super().__init__()
        # 1.2 搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 300)
        # 隐藏层2
        self.linear2 = nn.Linear(300, 300)
        # 隐藏层3
        self.linear3 = nn.Linear(300, 300)
        # 隐藏层4
        self.linear4 = nn.Linear(300, 300)
        # 输出层
        self.output = nn.Linear(300, output_dim)

    # 2、定义前向传播方法
    def forward(self, x):
        # 2.1 隐藏层1：加权求和 + 激活函数(relu)
        # x = self.linear1(x)
        # x = torch.relu(x)
        # 可以合并起来写：
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2：加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        # 2.3 输出层：加权求和 + 激活函数(softmax) → 这里只需要做 加权求和，因为在损失函数CrossEntropyLoss()中已经包含了softmax
        # 正常写法，但是不需要，后续用 多分类交叉熵损失函数CrossEntropyLoss()替代
        # CrossEntropyLoss() = softmax() + 损失计算
        # x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)
        # 2.4 返回结果
        return x


# todo 3:训练神经网络
def train(train_dataset, input_dim, output_dim):
    # 设置随机种子，确保结果可复现
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1、创建数据加载器，流程：数据 → 张量Tensor → 数据集TensorDataSet → 数据加载器DataLoader
    # 参1：数据集对象（1600条），参2：每批次的数据条数，参3：是否打乱数据（训练集：打乱，测试集：不打乱）
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)  # 显式指定生成器在cuda上
    # 2、创建神经网络模型
    model = estimateNet(input_dim=input_dim, output_dim=output_dim).to(device)
    # 3、定义损失函数，这里用的是：SmoothL1Loss()
    criterion = nn.SmoothL1Loss()
    # 4、创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 5、模型训练
    # 5.1 定义变量，记录训练的总轮数
    epochs = 25000
    # 5.2 开始（每轮的）训练
    for epoch in range(epochs):
        # 5.2.1 定义变量，记录每次训练的损失值，训练批次数
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量，表示训练开始的时间
        start_time = time.time()
        # 5.2.3 开始本轮的 每个批次的训练
        for x, y in train_loader:
            # 5.2.4 切换模型（状态）
            model.train()  # 训练模式  model.eval()    # 测试模式
            # 5.2.4 把数据移动到cuda上
            x, y = x.to(device), y.to(device)
            # 5.2.5 模型预测
            y_pred = model(x)
            # 5.2.6 计算损失
            loss = criterion(y_pred, y)
            # 5.2.7 梯度清零，反向传播，优化参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.8 累加损失值
            total_loss += loss.item()  # 把本轮的每批次（16条）的平均损失累加起来 第1批次的平均损失 + 第2批次的平均损失 + ...
            batch_num += 1
        # 5.2.9 至此，本轮训练结束，打印训练信息
        print(f"第{epoch + 1}轮 time:{time.time() - start_time:.2f} 损失值：{total_loss / batch_num:.4f}")
    # 6 走到这里，说明多轮训练结束，保存模型（参数）
    # 参1：模型对象的参数（权重矩阵，偏置矩阵） 参2：模型保存的文件名
    # print(f"\n保存的模型参数为：{model.state_dict()}\n")
    torch.save(model.state_dict(), f"./model/estimate_model.pth")  # 后缀名用:pth,pkl,pickle即可


# todo 4:测试神经网络
def evaluate(test_dataset, input_dim, output_dim):
    # 1、创建神经网络分类对象
    model = estimateNet(input_dim=input_dim, output_dim=output_dim)
    # 2、加载模型参数
    model.load_state_dict(torch.load(f"./model/estimate_model.pth"))

    # 3、创建测试集的数据加载器对象
    # 参1：数据集对象（1600条），参2：每批次的数据条数，参3：是否打乱数据（训练集：打乱，测试集：不打乱）
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)  # 显式指定生成器在cuda上
    # 4、定义变量，记录预测正确的样本个数
    correct = 0
    # 定义变量，记录测试集上的MAE和MSE
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    # 5、从数据加载器中，获取到每批次的数据
    for x, y in test_loader:
        # 5.1 切换模型状态 → 测试模式
        model.eval()
        # print(f"\nx:{x}  y:{y}\n")
        # 5.2 模型预测
        y_pred = model(x)
        # 5.3 根据加权求和，得到类别，用argmax()方法获取最大值对应的坐标，就是类别
        # 为什么要用argmax()而不用softmax()方法呢？因为在训练时，多分类交叉熵损失函数已经做了softmax()，
        # 所以我们的神经网络就没有添加softmax()方法，这就导致这里需要加argmax()方法来实现类似的操作
        mae = torch.nn.functional.l1_loss(y_pred, y, reduction='mean')  # dim=1 表示逐行处理
        mse = torch.nn.functional.mse_loss(y_pred, y, reduction='mean')  # dim=1 表示逐行处理
        # print(f"y_pred:{y_pred}")  # [第1条数据的预测分类,...]
        # print(f"y:{y}")
        # 5.4 统计预测正确的样本个数 其中，y_pred == y 表示：判断预测结果和真实标签是否相等，返回一个布尔值列表，其实就是布尔索引
        # correct += (y_pred == y).sum()
        total_mae += mae.item()
        total_mse += mse.item()
        total_samples += x.size(0)
    # 计算平均MAE和MSE
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    # 打印结果
    print(f"测试结果:")
    print(f"平均绝对误差 (MAE): {avg_mae:.6f}")
    print(f"均方误差 (MSE): {avg_mse:.6f}")
    print(f"均方根误差 (RMSE): {avg_mse ** 0.5:.6f}")
    
    # 6 走到这里，模型预测结束，打印准确率即可
    # print(f"准确率(Accuracy)：{correct / len(test_dataset):.4f}")


if __name__ == '__main__':
    # 1、创建数据集，获取数据集的输入特征维度和输出标签维度
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    print(f"训练集对象：{train_dataset}")
    print(f"测试集对象：{test_dataset}")
    print(f"输入特征数：{input_dim}")
    print(f"输出标签数：{output_dim}\n")
    # 2、创建神经网络模型
    # model = phonePriceNet(input_dim=input_dim, output_dim=output_dim)
    # 计算模型参数
    # 参1：模型对象 参2：输入数据的形状（批次大小，输入特征数），每批16条，每条20列特征
    # 输出的内容为：每层中参数的个数，例如：输入特征数是20，隐藏层1有128个神经元，那么参数个数为21*128=2688，因为还有一个偏置值
    # summary(model, input_size=(16, input_dim))

    # 3、模型训练
    # train(train_dataset, input_dim, output_dim)

    # 4、模型测试
    evaluate(test_dataset, input_dim, output_dim)
