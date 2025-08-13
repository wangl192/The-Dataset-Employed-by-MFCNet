#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-
from __future__ import division, print_function

import argparse
import torch
import torch.nn.init
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import pickle  # Add this import

from Utils import str2bool
import torch.nn as nn

import torch.utils.data.dataloader
import lmdb
import pyarrow as pa
import torch.utils.data as data
import os.path as osp

# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet_Dense')
# Model options
# ./data/models/gf3_ge1/checkpoint_2.pth
parser.add_argument('--resume', default=r'C:\Users\WL\Desktop\models\Mview_tst\checkpoint_9.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--extent-pos', default=1, type=int,
                    help='Extent of positive samples on the ground truth map')
parser.add_argument('--test-only', type=bool, default=False,
                    help='')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 10.0. Yes, ten is not typo)')  # 0.01 for sgd
parser.add_argument('--suffix', default='Mview_tst/',
                    help='suffix of trained modal')
parser.add_argument('--dataroot', type=str,
                    default=r'D:\deep_learning_data\WHU_SEN_lmdb',
                    help='path to dataset')
parser.add_argument('--name-train', type=str, default='train',
                    help='name of training dataset')
parser.add_argument('--name-test', type=str, default='test',
                    help='name of testing dataset')
parser.add_argument('--offset-test', type=int, default=6,
                    help='Offset value between test image pairs')
parser.add_argument('--enable-logging', type=str2bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='D:/deep_learning_data/deep_learning_methods/Net_train_test/data/logs/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='C:/Users/WL/Desktop/models/',
                    help='folder to output model checkpoints')
parser.add_argument('--search-rad', default=32, type=int,
                    help='Search radius for fft match')
parser.add_argument('--num-workers', default=0, type=int,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory', type=bool, default=True,
                    help='')
parser.add_argument('--mean-image', type=float, default=0.4309,
                    help='mean of train dataset for normalization')
parser.add_argument('--std-image', type=float, default=0.2236,
                    help='std of train dataset for normalization')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=40, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=4, metavar='BS',
                    help='input batch size for training (default: 1024)')  # 5
parser.add_argument('--test-batch-size', type=int, default=10, metavar='BST',
                    help='input batch size for testing (default: 1024)')  # 5
parser.add_argument('--freq', type=float, default=10.0,
                    help='frequency for cyclic learning rate')
parser.add_argument('--lr-decay', default=1e-6, type=float, metavar='LRD',
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='adam', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
# Device options
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--log-interval', type=int, default=10, metavar='LI',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = args.use_cuda and torch.cuda.is_available()

print(("NOT " if not args.cuda else "") + "Using cuda")

# 检查是否使用 cuda 加速
if args.cuda:
    cudnn.benchmark = True
    # 为所有的 GPU 设备设置随机种子，以便在使用 cuda 时进行随机数生成
    torch.cuda.manual_seed_all(args.seed)
# 设置 cudnn 为确定性模式
# 这意味着使用相同的输入和模型时，每次运行结果都将相同
# 但可能会降低性能，因为它会限制 cudnn 选择算法的灵活性
torch.backends.cudnn.deterministic = True

# create loggin directory
# 检查 args.log_dir 所指向的路径是否存在
if not os.path.exists(args.log_dir):
    # 如果路径不存在，使用 os.makedirs 函数创建该目录
    # 这将创建多层目录结构，例如，如果路径是 "a/b/c"，而 "a" 和 "b" 也不存在，会依次创建 "a"、"b" 和 "c"
    os.makedirs(args.log_dir)

# set random seeds
# 使用 random 模块的 seed 函数设置随机种子
random.seed(args.seed)
# 使用 torch 模块的 manual_seed 函数设置随机种子
torch.manual_seed(args.seed)
# 使用 numpy 模块的 random.seed 函数设置随机种子
np.random.seed(args.seed)


def loads_pickle(buf):
    """
    此函数用于将序列化的数据反序列化。

    Args:
        buf: 序列化的数据。
    Returns:
        反序列化后的数据。
    """
    if buf is None:
        return None
    return pickle.loads(buf)


class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, db_name, transform=None):

        self.full_path = osp.join(db_path, db_name)
        print(self.full_path)
        self.transform = transform

        self.env = lmdb.open(self.full_path, subdir=osp.isdir(self.full_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = loads_pickle(txn.get(b'__len__'))
            self.keys = loads_pickle(txn.get(b'__keys__'))
        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        img_pair = loads_pickle(byteflow)
        # 转换为 float32 类型
        img_sar_quart = img_pair[0]
        img_opt_quart = img_pair[1]

        # 如果是三通道光学图像，转换为单通道
        if len(img_opt_quart.shape) == 3:
            # 使用 RGB 转灰度的标准公式: Y = 0.299R + 0.587G + 0.114B
            img_opt_quart = np.dot(img_opt_quart[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

        # 如果定义了 transform 函数，对图像进行变换
        if self.transform:
            img_sar_quart = self.transform(img_sar_quart)
            img_opt_quart = self.transform(img_opt_quart)

        return img_sar_quart, img_opt_quart

    def __len__(self):
        """
        获取数据集的长度。
        """
        return self.length


def inspect_lmdb(db_path, db_name):
    """
    检查 LMDB 数据库的内容和状态
    """
    full_path = osp.join(db_path, db_name)
    print(f"\n正在检查 LMDB 数据库: {full_path}")

    if not osp.exists(full_path):
        print(f"错误: 数据库路径不存在: {full_path}")
        return False

    try:
        env = lmdb.open(full_path,
                        subdir=osp.isdir(full_path),
                        readonly=True,
                        lock=False)

        with env.begin() as txn:
            cursor = txn.cursor()
            print("\n数据库内容:")
            key_count = 0
            for key, _ in cursor:
                key_count += 1
                if key in [b'__len__', b'__keys__']:
                    print(f"找到特殊键: {key}")
            print(f"总键数: {key_count}")

            # 特别检查必需的键
            len_data = txn.get(b'__len__')
            keys_data = txn.get(b'__keys__')

            if len_data is None:
                print("警告: 未找到 '__len__' 键")
            if keys_data is None:
                print("警告: 未找到 '__keys__' 键")

        env.close()
        return len_data is not None and keys_data is not None

    except Exception as e:
        print(f"检查数据库时出错: {str(e)}")
        return False


def create_loaders():
    """
    此函数用于创建训练集和测试集的数据加载器。
    """
    # 检查训练和测试数据库
    train_db_ok = inspect_lmdb(args.dataroot, args.name_train)
    test_db_ok = inspect_lmdb(args.dataroot, args.name_test)

    if not (train_db_ok and test_db_ok):
        print("\n请确保已使用 madedb.py 正确创建数据库，并检查以下设置:")
        print(f"数据库根目录: {args.dataroot}")
        print(f"训练数据库名: {args.name_train}")
        print(f"测试数据库名: {args.name_test}")
        raise RuntimeError("数据库检查失败")

    # 根据是否使用 cuda 决定是否使用多进程和内存锁定
    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}
    # 定义数据变换，将数据转换为张量并进行归一化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean_image,), (args.std_image,)),
    ])
    # 创建训练集的数据加载器
    train_loader = torch.utils.data.DataLoader(
        # 使用 DatasetLMDB 类作为数据集，传入数据库路径、数据库名称和变换函数
        DatasetLMDB(db_path=args.dataroot,
                    db_name=args.name_train,
                    transform=transform),
        # 设置批量大小
        batch_size=args.batch_size,
        # 打乱数据顺序
        shuffle=True,
        # 传递额外的参数，如 num_workers 和 pin_memory
        **kwargs
    )
    # 创建测试集的数据加载器
    test_loader = torch.utils.data.DataLoader(
        # 使用 DatasetLMDB 类作为数据集，传入数据库路径、数据库名称和变换函数
        DatasetLMDB(db_path=args.dataroot,
                    db_name=args.name_test,
                    transform=transform),
        # 设置测试集的批量大小
        batch_size=args.test_batch_size,
        # 不打乱数据顺序
        shuffle=False,
        # 传递额外的参数，如 num_workers 和 pin_memory
        **kwargs
    )
    # 返回训练集和测试集的数据加载器
    return train_loader, test_loader


def fft_match_batch(feature_sar, feature_opt, search_rad):
    """
    此函数用于对 SAR 和 OPT 特征进行基于 FFT 的匹配处理。

    Args:
        feature_sar: SAR 特征数据，是一个张量。
        feature_opt: OPT 特征数据，是一个张量。
        search_rad: 搜索半径，用于确定匹配的范围。
    """
    # 将搜索半径转换为整数
    search_rad = int(search_rad)
    # 获取 SAR 特征的形状
    b, c, w, h = np.shape(feature_sar)
    # 计算一个与搜索半径相关的变量
    nt = search_rad
    # 创建一个与 SAR 特征形状相同的全零张量 T
    T = torch.zeros(np.shape(feature_sar))
    # 将 T 的部分区域置为 1，形成一个特定的模式
    T[:, :, 0:h - 2 * nt, 0:w - 2 * nt] = 1
    # 创建一个与 SAR 特征形状相同的全零虚部张量
    fake_imag = torch.zeros(np.shape(feature_sar))
    # 如果使用 CUDA，将 T 和 fake_imag 转移到 GPU 上
    if args.use_cuda:
        T = T.cuda()
        fake_imag = fake_imag.cuda()
    # 计算 SAR 特征的平方
    sen_x = feature_sar ** 2
    # 对 sen_x 进行二维傅里叶变换
    tmp1 = torch.fft.fft2(sen_x)
    # 对 T 进行二维傅里叶变换
    tmp2 = torch.fft.fft2(T)
    # 计算 tmp1 和 tmp2 的共轭相乘并求和
    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)
    # 对 tmp_sum 进行二维逆傅里叶变换
    ssd_f_1 = torch.fft.ifft2(tmp_sum)
    # 取 ssd_f_1 的实部
    ssd_fr_1 = torch.real(ssd_f_1)
    # 截取 ssd_fr_1 的一部分区域
    ssd_fr_1 = ssd_fr_1[:, 0:2 * nt + 1, 0:2 * nt + 1]
    # 从 feature_opt 中截取一个参考区域
    ref_T = feature_opt[:, :, nt:w - nt, nt:h - nt]
    # 创建一个全零张量 ref_Tx
    ref_Tx = torch.zeros(np.shape(feature_opt))
    # 将 ref_T 放置在 ref_Tx 的部分区域
    ref_Tx[:, :, 0:w - 2 * nt, 0:h - 2 * nt] = ref_T
    # 如果使用 CUDA，将 ref_Tx 转移到 GPU 上
    if args.use_cuda:
        ref_Tx = ref_Tx.cuda()
    # 对 feature_sar 进行二维傅里叶变换
    tmp1 = torch.fft.fft2(feature_sar)
    # 对 ref_Tx 进行二维傅里叶变换
    tmp2 = torch.fft.fft2(ref_Tx)
    # 计算 tmp1 和 tmp2 的共轭相乘并求和
    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)
    # 对 tmp_sum 进行二维逆傅里叶变换
    ssd_f_2 = torch.fft.ifft2(tmp_sum)
    # 取 ssd_f_2 的实部
    ssd_fr_2 = torch.real(ssd_f_2)
    # 截取 ssd_f_2 的一部分区域
    ssd_fr_2 = ssd_fr_2[:, 0:2 * nt + 1, 0:2 * nt + 1]
    # 计算最终的 ssd_batch 结果
    ssd_batch = (ssd_fr_1 - 2 * ssd_fr_2) / w / h
    return ssd_batch


def loss_fft_match_batch(out_sar, out_opt, gt_map, search_rad,i=0,j=0):
    bs = out_sar.shape[0]
    # 如果批量大小不等于预设的 batch_size，则返回 0
    if bs != args.batch_size:
        return 0
    # 调用 fft_match_batch 函数对 out_sar 和 out_opt 进行处理
    out = fft_match_batch(out_sar, out_opt, search_rad)

    batch_0 = out[0]  # 形状为 (w, h)
    # 计算最大值和最小值
    max_value = torch.max(batch_0)
    min_value = torch.min(batch_0)

    # 打印结果
    print(f"第1个batch的最大值: {max_value.item()}")
    print(f"第1个batch的最小值: {min_value.item()}")
    #对 out 进行变换，将其元素取反并加 1
    # out = 1 - out;
    # if not args.test_only:
    #     A = out[:, i+args.search_rad, j+args.search_rad]
    # else:
    #     A = out[:,args.offset_test+args.search_rad,args.offset_test+args.search_rad]
    # # 定义 gamma 和 margin 常量

    gamma = 32  # 32
    margin = 0.25
    margin_bj = 0.3

    ##########
    # 计算 gt_map 的补集
    gt_map_neg = 1 - gt_map
    # 将 gt_map 转换为布尔型张量
    gt_mapx = gt_map.type(dtype=torch.bool)
    # 根据 gt_mapx 从 out 中筛选元素存储在 sp 中
    sp = out[gt_mapx]
    # 将 gt_map_neg 转换为布尔型张量
    gt_map_negx = gt_map_neg.type(dtype=torch.bool)
    # 根据 gt_map_negx 从 out 中筛选元素存储在 sn 中
    sn = out[gt_map_negx]
    # 重塑 sp 的形状
    sp = sp.view(out.size()[0], -1)
    # 重塑 sn 的形状
    sn = sn.view(out.size()[0], -1)
    ###########



    # Ap_expanded = A.unsqueeze(1).expand(-1, sp.size(1))
    # An_expanded = A.unsqueeze(1).expand(-1, sn.size(1))
    # 计算 ap，对 -sp.detach() + 1 + margin 进行裁剪，使其最小为 0
    # 计算 an，对 sn.detach() + margin 进行裁剪，使其最小为 0
    ap = torch.clamp_min(sp.detach() - margin + margin_bj, min=0.)
    an = torch.clamp_min(-sn.detach() + (1 - margin) + margin_bj, min=0.)

    # ap = 0.5+torch.clamp(sp.detach() - Ap_expanded, min=0.)
    # an = 0.25+torch.clamp((1 - An_expanded) - sn.detach(), min=0.)

    # 定义 delta_p 和 delta_n 常量
    delta_p = margin + 1
    delta_n = 1 - margin

    # 计算 logit_p
    # logit_p = -ap * (sp - delta_p) * gamma
    # # 计算 logit_n
    # logit_n = an * (sn - delta_n) * gamma

    logit_p = ap * (sp - delta_p) * gamma
    # 计算 logit_n
    logit_n = an * (-sn + delta_n) * gamma
    # 创建 Softplus 激活函数实例
    soft_plus = nn.Softplus()
    # 计算 loss_circle
    loss_circle = soft_plus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p, dim=1))
    # 计算 loss_regu，取 loss_circle 的平均值
    loss_regu = loss_circle.mean()
    return loss_regu


def train(train_loader, model, gt_maps, optimizer, epoch, logger):
    """
    此函数用于训练模型。

    Args:
        train_loader: 训练数据加载器，提供训练数据。
        model: 待训练的模型。
        gt_maps: 真实映射列表。
        optimizer: 优化器，用于更新模型参数。
        epoch: 当前的训练轮次。
        logger: 日志记录器，用于记录训练信息。
    """
    # 将模型设置为训练模式
    model.train()
    # 使用 tqdm 包装 train_loader 以显示进度条
    pbar = tqdm(enumerate(train_loader))
    # 创建一个与 gt_maps[0] 形状相同的全零张量
    gt_map_rnd = torch.zeros_like(gt_maps[0])
    # 如果使用 CUDA，将 gt_map_rnd 移动到 GPU 上
    if args.use_cuda:
        gt_map_rnd = gt_map_rnd.cuda()
    # 遍历训练数据
    for batch_idx, data in pbar:
        # 从数据中分离出 SAR 和 OPT 数据
        data_sar, data_opt = data
        # print(data_sar.shape)
        # print(data_opt.shape)
        # 裁剪 data_sar
        data_sar = data_sar[:, :, 24:232, 24:232]
        # 生成随机偏移量 i 和 j
        i = random.randint(-15, 15)
        j = random.randint(-15, 15)
        # 对 data_opt 进行裁剪和偏移
        data_opt = data_opt[:, :, 24 + i:232 + i, 24 + j:232 + j]
        # 重置 gt_map_rnd 为全零
        gt_map_rnd.fill_(0)
        # 根据偏移量和范围设置 gt_map_rnd 的部分区域为 1
        gt_map_rnd[:, args.search_rad - args.extent_pos + i:args.search_rad + args.extent_pos + 1 + i,
        args.search_rad - args.extent_pos + j:args.search_rad + args.extent_pos + 1 + j] = 1
        # 如果使用 CUDA，将 data_sar 和 data_opt 移动到 GPU 上
        if args.cuda:
            data_sar, data_opt = data_sar.cuda(), data_opt.cuda()
        # 通过模型得到输出
        out_opt, out_sar, attn_opt, attn_sar = model(data_opt, data_sar)
        # 计算损失
        loss = loss_fft_match_batch(out_sar, out_opt, gt_map_rnd, args.search_rad,i,j)
        # 如果损失不为 0
        if loss != 0:
            # 清空梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 调整学习率
            adjust_learning_rate(optimizer)
            # 记录训练信息
            if batch_idx % args.log_interval == 0:
                pbar.set_description(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data_sar), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader),
                        loss.item()))
    # 如果启用日志记录，记录损失值
    if (args.enable_logging):
        logger.log_value('loss', loss.item()).step()
    suffix = args.suffix
    try:
        # 检查目录是否存在
        os.stat('{}{}'.format(args.model_dir, suffix))
    except:
        # 若不存在则创建目录
        os.makedirs('{}{}'.format(args.model_dir, suffix))
    # 保存模型检查点
    torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
               '{}{}/checkpoint_{}.pth'.format(args.model_dir, suffix, epoch))


def test(test_loader, model, gt_maps, epoch, logger):
    """
    此函数用于测试模型。

    Args:
        test_loader: 测试数据加载器，提供测试数据。
        model: 待测试的模型。
        gt_maps: 真实映射列表。
        epoch: 当前的训练轮次（可能用于日志记录）。
        logger: 日志记录器，用于记录测试信息。
    """
    # 将模型设置为评估模式
    model.eval()
    # 存储测试结果
    results = []
    pos_max = []
    # 重塑 gt_maps[1][0] 为向量
    gt_map_vec = torch.reshape(gt_maps[1][0], [1, (2 * args.search_rad + 1) * (2 * args.search_rad + 1)])
    # 使用 tqdm 包装 test_loader 以显示进度条
    pbar = tqdm(enumerate(test_loader))
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据
        for batch_idx, (data_sar, data_opt) in pbar:
            # 裁剪 data_sar
            data_sar = data_sar[:, :, 24:232, 24:232]
            # 对 data_opt 进行裁剪和偏移
            data_opt = data_opt[:, :, 24 + args.offset_test:232 + args.offset_test,
                       24 + args.offset_test:232 + args.offset_test]
            # 如果使用 CUDA，将 data_sar 和 data_opt 移动到 GPU 上
            if args.cuda:
                data_sar, data_opt = data_sar.cuda(), data_opt.cuda()
            # 通过模型得到输出
            out_opt, out_sar, attn_opt, attn_sar = model(data_opt, data_sar)
            # 调用 fft_match_batch 函数处理 out_sar 和 out_opt
            out = fft_match_batch(out_sar, out_opt, args.search_rad)
            # 重塑 out 为向量
            out_vec = torch.reshape(out, [out.shape[0], (2 * args.search_rad + 1) * (2 * args.search_rad + 1)])
            # 获取 out_vec 每行的最小元素的索引
            out_min = out_vec.min(1)[1]
            # 遍历每个样本
            for i in range(out.shape[0]):
                # 获取结果并移动到 CPU
                rst = gt_map_vec[0, out_min[i]].cpu().item()  # 修改此处
                # 存储结果
                results.append(rst)
                # 存储最小值的索引
                pos_max.append(out_min[i].cpu().item())  # 修改此处

    # 计算正确率
    correct_rate = np.sum(results) / len(results)
    print('Validation Results: ', correct_rate)
    return correct_rate


def adjust_learning_rate(optimizer):
    """
    此函数用于调整优化器的学习率。

    Args:
        optimizer: 优化器，用于更新模型参数。
    """
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    # 遍历优化器的参数组
    for group in optimizer.param_groups:
        # 如果参数组中没有'step'键，初始化为 0
        if 'step' not in group:
            group['step'] = 0.
        # 否则'step'加 1
        else:
            group['step'] += 1.
        # 按照公式更新学习率
        group['lr'] = args.lr * (
                1.0 - float(group['step']) * float(4 * args.batch_size) / (
                233244 * 4 * float(args.epochs)))  # 63338 for spring, 233244 for all
    return


def create_optimizer(model, new_lr):
    """
    此函数用于创建优化器。

    Args:
        model: 待优化的模型。
        new_lr: 初始学习率。
    """
    # 根据优化器类型创建优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    # 不支持的优化器类型将引发异常
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, test_loaders, model, gt_map, logger, file_logger):
    """
    主函数，用于协调训练和测试流程。

    Args:
        train_loader: 训练数据加载器。
        test_loaders: 测试数据加载器列表或单个测试数据加载器。
        model: 待训练和测试的模型。
        gt_map: 真实映射。
        logger: 日志记录器，用于记录训练和测试信息。
        file_logger: 文件日志记录器，用于记录配置信息等。
    """
    # 打印实验配置
    print('\nparsed options:\n{}\n'.format(vars(args)))
    # 若启用日志记录，将配置信息记录到文件中
    # if (args.enable_logging):
    #    file_logger.log_string('logs.txt', '\nparsed options:\n{}\n'.format(vars(args)))
    # 如果使用 CUDA，将模型移动到 GPU 上
    if args.cuda:
        model.cuda()
    # 创建优化器
    optimizer1 = create_optimizer(model, args.lr)
    # 若从检查点恢复
    if args.resume:
        # 检查检查点文件是否存在
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            # 加载检查点
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
    # 获取起始和结束轮次
    start = args.start_epoch
    end = start + args.epochs
    test_correct_rates = []
    # 进行多个轮次的训练和测试
    for epoch in range(start, end):
        # 如果不是仅测试
        if not args.test_only:
            # 进行训练
            train(train_loader, model, gt_map, optimizer1, epoch, logger)
        # 进行测试
        correct_rate =test(test_loaders, model, gt_map, epoch, logger)
        test_correct_rates.append(correct_rate)
        with open('C:/Users/WL/Desktop/models/record.txt', 'a') as f:
            f.write(f'Epoch {epoch}: {correct_rate}\n')


if __name__ == '__main__':
    """
    程序的入口点，进行各种初始化和调用主要的训练和测试流程。
    """
    # 获取日志目录
    LOG_DIR = args.log_dir
    # 如果日志目录不存在，创建目录
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    # 拼接日志目录和后缀
    LOG_DIR = os.path.join(args.log_dir, args.suffix)
    # 初始化日志记录器为 None
    logger, file_logger = None, None
    # 导入模型类并创建模型实例
    #
    # from SFcNet import SFcNetPseudo
    # model = SFcNetPseudo()

    # from OSMNet import SSLCNetPseudo
    # model = SSLCNetPseudo()

    from Mul_net import Mul_OSnetPseudo
    model = Mul_OSnetPseudo()
    # 打印模型参数的类型和大小
    # for param in model.parameters():
    #     print(type(param.data), param.size())
    # 若启用日志记录，导入日志记录器并创建实例
    # if (args.enable_logging):
    #     from Loggers import Logger, FileLogger
    #     logger = Logger(LOG_DIR)
    #     # file_logger = FileLogger(./log/+suffix)
    # 创建全零的 gt_map 张量
    gt_map = torch.zeros(args.batch_size, 2 * args.search_rad + 1, 2 * args.search_rad + 1)
    # 将 gt_map 的部分区域置为 1
    gt_map[:, args.search_rad - args.extent_pos:args.search_rad + args.extent_pos + 1,
    args.search_rad - args.extent_pos:args.search_rad + args.extent_pos + 1] = 1
    # 创建偏移后的 gt_map_shift 张量
    offset_zh = args.offset_test
    gt_map_shift = torch.zeros(args.batch_size, 2 * args.search_rad + 1, 2 * args.search_rad + 1)
    gt_map_shift[:, args.search_rad - args.extent_pos + offset_zh:args.search_rad + args.extent_pos + 1 + offset_zh,
    args.search_rad - args.extent_pos + offset_zh:args.search_rad + args.extent_pos + 1 + offset_zh] = 1
    # 如果使用 CUDA，将 gt_map 和 gt_map_shift 移到 GPU 上
    if args.use_cuda:
        gt_map = gt_map.cuda()
        gt_map_shift = gt_map_shift.cuda()
    # 创建训练和测试数据加载器
    train_loader, test_loaders = create_loaders()
    # 将 gt_map 和 gt_map_shift 存储在列表中
    gt_maps = [gt_map, gt_map_shift]
    # 调用 main 函数进行训练和测试
    main(train_loader, test_loaders, model, gt_maps, logger, file_logger)
