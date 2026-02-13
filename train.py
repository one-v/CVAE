import os
import time
import torch
from torch.utils.data import DataLoader
from utils import create_dataset, model_list, get_args

from models import VAE

# 解决 OpenMP 库重复初始化问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 日志文件路径
LOG_FILE = './training_log.txt'

# 确保日志文件存在
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write(f"训练日志开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")


def log_message(message):
    """记录日志信息到txt文件"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}\n"

    # 打印到控制台
    print(message)

    # 写入日志文件
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def train(args, train_dataset):
    # 设置随机种子，确保结果可复现
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建数据集加载器，获取数据集的输入特征维度和输出标签维度
    data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    # 定义损失函数，使用 Smooth L1  + KLD 作为最终损失函数
    def loss_fn(recon_x, x, mean, log_var):
        # 使用 Smooth L1  + KLD 作为最终损失函数
        SmoothL1 = torch.nn.functional.smooth_l1_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return (SmoothL1 + KLD) / x.size(0)

    # 初始化模型，使用自定义 VAE 模型
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    # 选择 Adam 优化器
    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    log_loss = 0.0
    log_batch_num = 0
    log_start_time = time.time()
    # 开始训练
    for epoch in range(args.epochs):
        # 定义变量，记录每次训练的损失值，训练批次数
        total_loss, batch_num = 0.0, 0
        # 定义变量，表示训练开始的时间
        start_time = time.time()
        for iteration, (x, y) in enumerate(data_loader):
            # 切换模型（状态）
            vae.train()
            x, y = x.to(device), y.to(device)
            # 判断使用VAE还是CVAE
            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # 计算损失值
            loss = loss_fn(recon_x, x, mean, log_var)

            # 梯度清零、反向传播、参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累加损失值：把本轮的每批次的平均损失累加起来 第1批次的平均损失 + 第2批次的平均损失 + ...
            total_loss += loss.item()
            batch_num += 1
            log_loss += loss.item()
            log_batch_num += 1

        # 计算本轮训练的平均损失和时间
        avg_loss = total_loss / batch_num
        time_cost = time.time() - start_time

        # 至此，本轮训练结束，打印训练信息
        epoch_info = f"第{epoch + 1}轮\t\ttime:{time_cost:.2f}\t\t损失值：{avg_loss:.2f}"
        print(epoch_info)

        # 每100轮记录一次详细日志
        if (epoch + 1) % 100 == 0:
            detailed_info = f"=== 详细记录 === 当前轮数: {epoch + 1}的前100轮的平均损失值: {log_loss / log_batch_num:.2f}, 平均耗时: {time.time() - log_start_time:.2f}秒"
            # 重置日志损失值和批次数和开始时间
            log_loss = 0.0
            log_batch_num = 0
            log_start_time = time.time()
            log_message(detailed_info)

    # 记录模型完成
    model_info = f"模型训练完成: CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}"
    log_message(model_info)

    # 多轮训练结束，保存模型（参数） 后缀名用:pth,pkl,pickle即可
    # 参1：模型对象的参数（权重矩阵，偏置矩阵） 参2：模型保存的文件名
    model_path = f"./model/CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}.pth"
    torch.save(vae.state_dict(), model_path)
    log_message(f"模型已保存到: {model_path}")


def evaluate(args, test_dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 初始化自定义 VAE 模型
    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=6 if args.conditional else 0).to(device)

    # 创建数据集加载器，获取数据集的输入特征维度和输出标签维度
    data_loader = DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    # 加载模型参数
    model_path = f"./model/CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}.pth"
    vae.load_state_dict(torch.load(model_path, weights_only=True))
    # 3、切换模型（状态）
    vae.eval()

    # 定义变量，记录测试集上的MAE和MSE
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0

    # 遍历测试集
    with torch.no_grad():
        for iteration, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, y)
            else:
                recon_x, mean, log_var, z = vae(x)

            # 计算MAE和MSE
            mae = torch.nn.functional.l1_loss(recon_x, x, reduction='sum')
            mse = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')

            total_mae += mae.item()
            total_mse += mse.item()
            total_samples += x.size(0)

    # 计算平均MAE和MSE
    avg_mae = total_mae / total_samples
    avg_mse = total_mse / total_samples

    # 打印结果
    log_message(f"测试结果:")
    log_message(f"平均绝对误差 (MAE): {avg_mae:.6f}")
    log_message(f"均方误差 (MSE): {avg_mse:.6f}")
    log_message(f"均方根误差 (RMSE): {avg_mse ** 0.5:.6f}")


if __name__ == '__main__':
    # 1、创建数据集，获取数据集的输入特征维度和输出标签维度
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # 2、获取模型列表
    train_list = model_list()
    # 3、定义变量，记录当前状态是训练还是测试
    state_train_test = True
    # 4、遍历模型列表，批量训练模型
    for train_dict in train_list:
        # 5、判断是否训练该模型
        if not train_dict["if_train"]:
            continue
        # 6、获取命令行参数
        args = get_args(train_dict)
        # 7、输出当前的训练模型名
        model_name = f"CVAE_{args.epochs}_{args.learning_rate}_{args.batch_size}_{args.encoder_layer_sizes}_{args.latent_size}_{args.decoder_layer_sizes}"
        log_message(f"开始训练模型：{model_name}")
        # 8、通过状态，判断当前是训练还是测试
        if state_train_test:
            train(args, train_dataset)
        else:
            evaluate(args, test_dataset)
    log_message("所有模型训练已完成") if state_train_test else log_message("所有模型测试已完成")
