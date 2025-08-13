#!/usr/bin/python2 -utt
# -*- coding: utf-8 -*-


from __future__ import division, print_function
import argparse
import torch 
import torch.nn.init
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt



# Training settings
parser = argparse.ArgumentParser(description='PyTorch HardNet_Dense')
# Model options
# parser.add_argument('--use-attention', type=bool, default=False,
#                     help='use the MultFca Attention layer or not')
parser.add_argument('--resume', default=r'C:\Users\WL\Desktop\models\MDDja_tst\checkpoint_2.pth', type=str, metavar='PATH',
                    help='path to trained model without the attention module(default: none)')
parser.add_argument('--resume-att', default='/models/checkpoint_osmnet.pth', type=str, metavar='PATH',
                    help='path to trained model with MultFca attention module (default: none)')
parser.add_argument('--extent-pos', default=1, type=int,
                    help='Extent of positive samples on the ground truth map')
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
# Device options
parser.add_argument('--use-cuda', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


args = parser.parse_args()

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = args.use_cuda and torch.cuda.is_available()

print(("NOT " if not args.cuda else "") + "Using cuda")

if args.cuda:
    cudnn.benchmark = True


def fft_match_batch(feature_sar, feature_opt, search_rad):
    search_rad = int(search_rad)
    b, c, w, h = np.shape(feature_sar)
    nt = search_rad
    T = torch.zeros(np.shape(feature_sar))
    T[:, :, 0:h - 2 * nt, 0:w - 2 * nt] = 1
    if args.use_cuda:
        T = T.cuda()

    sen_x = feature_sar ** 2

    tmp1 = torch.fft.fft2(sen_x)
    tmp2 = torch.fft.fft2(T)

    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)

    ssd_f_1 = torch.fft.ifft2(tmp_sum)

    ssd_fr_1 = torch.real(ssd_f_1)
    ssd_fr_1 = ssd_fr_1[:, 0:2 * nt + 1, 0:2 * nt + 1]

    ref_T = feature_opt[:, :, nt:w - nt, nt:h - nt]
    ref_Tx = torch.zeros(np.shape(feature_opt))
    ref_Tx[:, :, 0:w - 2 * nt, 0:h - 2 * nt] = ref_T

    if args.use_cuda:
        ref_Tx = ref_Tx.cuda()

    tmp1 = torch.fft.fft2(feature_sar)
    tmp2 = torch.fft.fft2(ref_Tx)

    tmp_sum = torch.sum(tmp1 * torch.conj(tmp2), 1)
    ssd_f_2 = torch.fft.ifft2(tmp_sum)
    ssd_fr_2 = torch.real(ssd_f_2)
    ssd_fr_2 = ssd_fr_2[:, 0:2 * nt + 1, 0:2 * nt + 1]
    ssd_batch = (ssd_fr_1 - 2 * ssd_fr_2) / w / h

    return ssd_batch


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    # from OSMNet import SSLCNetPseudo # SSLCNetPseudo_Att
    # model = SSLCNetPseudo()
    # from SFcNet import SFcNetPseudo
    # model = SFcNetPseudo()
    from Mul_net import Mul_OSnetPseudo
    model = Mul_OSnetPseudo()

    path_model = args.resume

    if os.path.isfile(path_model):
        print('=> loading checkpoint {}'.format(path_model))
        checkpoint = torch.load(path_model)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print('=> no checkpoint found at {}'.format(args.resume))

    # switch to evaluate mode
    model.eval()

    # 读取图像路径
    sar_image_path = './ims/1patches/patch_0.png'
    opt_image_path = './ims/patches/patch_0.png'

    # 检查文件是否存在
    if not os.path.exists(sar_image_path):
        print(f"错误: SAR 图像文件不存在: {sar_image_path}")
        exit(1)

    if not os.path.exists(opt_image_path):
        print(f"错误: 光学图像文件不存在: {opt_image_path}")
        exit(1)

    # 读取图像
    img_sar = cv2.imread(sar_image_path, 0)
    img_opt = cv2.imread(opt_image_path, 0)

    # 确保图像已正确读取
    if img_sar is None:
        print(f"无法读取 SAR 图像: {sar_image_path}")
        exit(1)

    if img_opt is None:
        print(f"无法读取光学图像: {opt_image_path}")
        exit(1)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((args.mean_image,), (args.std_image,))
    ])

    data_sar = transform(img_sar)
    data_opt = transform(img_opt)

    # 重塑为 (batch_size, channels, height, width)
    data_sar = torch.reshape(data_sar, (1, 1, 256, 256))
    data_opt = torch.reshape(data_opt, (1, 1, 256, 256))


    if args.cuda:
        data_sar, data_opt = data_sar.cuda(), data_opt.cuda()
        model.cuda()

    out_opt, out_sar = model(data_opt, data_sar)
    out = fft_match_batch(out_sar, out_opt, args.search_rad)
    out_s = torch.squeeze(out)
    out_s = out_s.cpu().detach().numpy()

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img_opt, cmap='viridis')  # 设置色带为 viridis
    plt.subplot(2, 2, 2)
    plt.imshow(img_sar, cmap='viridis')  # 设置色带为 viridis
    plt.subplot(2, 2, 3)
    plt.imshow(out_s, cmap='viridis')  # 设置色带为 viridis
    plt.show()

    # 再次创建一个新的图形窗口来保存 out_s 图像
    plt.figure()
    plt.imshow(out_s, cmap='viridis')

    # 隐藏坐标轴
    plt.axis('off')

    # 指定保存的目录
    save_directory = 'C:/Users/WL/Desktop/out'  # 替换为你实际想要保存的目录路径
    # 确保目录存在，如果不存在则创建
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 构建完整的保存路径
    save_path = os.path.join(save_directory, 'out_s_image.png')

    # 保存图像，设置分辨率为 600 dpi，去除白边
    plt.savefig(save_path, dpi=600, bbox_inches='tight', pad_inches=0)

    # 关闭当前图形窗口
    plt.close()






