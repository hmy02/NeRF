# -*- coding: gbk -*-

import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from myown_run_nerf_helpers import *

from myown_load_llff import load_llff_data  # 用于加载llff型数据
# from load_deepvoxels import load_dv_data
# from load_blender import load_blender_data
# from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """
    根据chunk的大小进行分区计算，避免内存溢出等问题，下文netchunk定义为1024*64
    :param fn: 网络模型
    :param chunk: 以多大的尺寸进行分块
    :return: 分区计算的结果
    """
    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024 * 64):
    """
    整合数据与视角方向，交给模型，输出output
    :param inputs: 采样点的3D坐标
    :param viewdirs: 视角方向
    :param fn: 网络模型
    :param embed_fn: 嵌入输入的坐标，映射到新的空间
    :param embeddirs_fn: 嵌入输入的视角方向，映射到新的空间
    :param netchunk: 分块
    :return: 输出
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]) # 铺平数据
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024 * 32, **kwargs):
    """

    :param rays_flat: 包含ray原点、ray方向、最大最小距离、方向单位向量
    :param chunk: 分块
    :param kwargs:
    :return: 分块计算后的结果
    """
    all_ret = {} # 存储每个块处理后的结果
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], **kwargs) # 体素渲染
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int 图像像素的高
      W: int 图像像素的宽
      focal: float 焦距
      chunk: int 分块
      rays: array of shape [2, batch_size, 3]. 每个batch的原点和方向分别用一个三维向量表示
      c2w: array of shape [3, 4]. 旋转矩阵
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. 射线最近距离.
      far: float or array of shape [batch_size]. 射线最远距离.
      use_viewdirs: bool. If True 在射线参数中加入观察方向
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. 预测的RGB
      disp_map: [batch_size]. 视差图
      acc_map: [batch_size]. 射线的不透明度alpha
      extras: render_rays()的返回值
    """
    # 如果有提供c2w旋转矩阵，即计算得到每一个像素点的相机原点，射线方向的世界坐标；否则直接使用提供的坐标
    if c2w is not None:
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        rays_o, rays_d = rays

    # 如果使用视角信息需要进行的一些单独处理
    if use_viewdirs:
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # viewdirs用c2w进行旋转，rays_d用c2w_staticcam进行旋转
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape
    if ndc:
        # 转换成ndc视角
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # 对射线坐标进行铺平
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    # 获得最近和最远距离的与rays_o,d的等长形式并进行拼接。
    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1) # 如果使用了viewdirs，向量是11维，否则8维

    # 分块运行render_rays进行体素渲染，获得三张map
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict] # 返回一个列表，其中包含提取出来的 RGB、视差、累积不透明度等信息


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    """

    :param render_poses: 相机位姿
    :param hwf: H, W, Focal
    :param K: 内参矩阵
    :param chunk: 分块
    :param render_kwargs: 一些其它在字典内定义了的参数
    :param gt_imgs: ground truth，用于渲染过程中评估模型性能
    :param savedir:
    :param render_factor: 控制渲染过程中的降采样因子的参数，默认为0
    :return: 渲染得到的 RGB 和视差图像
    """
    H, W, focal = hwf

    if render_factor != 0:
        # 降采样，会使运行速度更快
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        # 运行render得到三个map
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i == 0:
            print(rgb.shape, disp.shape)

        # 为提高效率禁用掉了
        """
        if gt_imgs is not None and render_factor==0:
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
            print(p)
        """

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    # 将渲染后的全部图像进行堆叠，并返回
    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps


def create_nerf(args):
    """
    实例化nerf
    :param args: 所有通过config_parser()定义的参数
    :return:
    """
    # 对x,y,z和方向信息都进行了位置编码，输入是x,y,z三维，输出是input_ch=63维；如果use_viewdirs为真，则input_ch_views=27维；
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4  # 输出的通道数
    skips = [4]  # 跳跃连接，一种保留底层特征的手段
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())  # 梯度

    # 如果有额外的精细样本数量
    model_fine = None
    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    # 模型output
    network_query_fn = lambda inputs, viewdirs, network_fn: run_network(inputs, viewdirs, network_fn,
                                                                        embed_fn=embed_fn,
                                                                        embeddirs_fn=embeddirs_fn,
                                                                        netchunk=args.netchunk)

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir  # 实验结果输出路径
    expname = args.expname  # 实验名称

    ##########################

    # 检查点(加载已有模型参数)
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # 如果不是llff-style的数据，不采用ndc
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    # 返回渲染所需的训练集，测试集的参数dict，训练开始节点，梯度参数，优化器参数
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """根据权重值，采样点的颜色进行加权平均，得到可视的rgb图像，视差，累计权重，权重值和深度等信息
    Args:
        raw: [num_rays, num_samples along ray, 4]. 模型预测结果
        z_vals: [num_rays, num_samples along ray]. 射线方向的积分时间(dt)
        rays_d: [num_rays, 3]. 射线方向
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    # 计算连续z值之间的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    # 计算每条光线的实际距离
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    # 如果指定了噪声就添加噪声
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # 如果用于测试目的，覆盖随机采样的数据
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # 映射透明度到01区间
    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    # 使用上述权重重新计算每个点的颜色
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    # 添加白色背景
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """
    主要是将各种数据处理成raw2output()所需的形式，并调用raw2output()进行体素渲染
    Args:
      ray_batch: array of shape [batch_size, ...]. 光线起点、方向、最小距离、最大距离、方向单位向量
      network_fn: function. NERF network
      network_query_fn: 将查询传递给network_fn
      N_samples: int. 采样点数
      retraw: bool. 是否压缩数据
      lindisp: bool. If True, 在深度图上逆向线性采样
      perturb: float, 0 or 1. 采样是否随机
      N_importance: int. 增加的精细采样点数
      network_fine: 精细网络
      white_bkgd: bool. If True, 白色背景
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    # 将数据提取出来
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)  # 0-1线性采样N_samples个值
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples]) # 扩展维度以匹配光线数

    # 加入扰动
    if perturb > 0.:
        # 获取样本间间隔
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # 分层随机采样
        t_rand = torch.rand(z_vals.shape)

        # 如果用于测试，用numpy的固定随机数覆盖u
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # o+td(采样点的3D坐标)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    #     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)  # 送进网络
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                 pytest=pytest)  # 体素渲染模型输出

    # fine网络，重复上述步骤，采用不同采样点(文章中提到的先进行粗采样，根据采样结果再进行精细采样(Nc+Nf))
    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        # 通过sample_pdf进行精细采样，并与原来粗采样的结果进行拼接
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                            None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        #         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd,
                                                                     pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    # 生成config.txt文件
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    # 指定实验名称
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    # 指定实验结果输出路径
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    # 输入数据的目录
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    # 每次梯度下降的随机射线数量
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    # 学习率衰减
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # 并行处理射线的数量
    parser.add_argument("--chunk", type=int, default=1024 * 32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    # 并行处理输入网络的点的数量
    parser.add_argument("--netchunk", type=int, default=1024 * 64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    # 每次只从一张图像中选取随机射线
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    # 不从保存的模型ckpt文件中载入权重
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    # 为coarse网络载入特定的权重文件
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    # 粗样本数量
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    # 精细样本数量
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    # 是否扰动
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # 使用5D替代3D信息(感觉实际上是6D?)
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    # 是否加入位置编码操作
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    # 位置编码操作对于3D位置信息的所升维数，默认L=10
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    # 位置编码操作对于2D方向信息的所升维数，默认L=4
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    # 加在规范化不透明度输出sigma上的噪声的标准差
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    # 只渲染(只向前传播)
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    # 测试集代替render_poses路径
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    # 下采样因子以加快渲染速度，一般设为4/8用于快速预览
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    # 中心crops上训练的迭代次数
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    # 三种数据类型: llff, blender, deepvoxels
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    # 测试集与验证集加载数据的比例，分别为1:N，对于大型数据集很有效
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    # 图像下采样率
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    # 是否使用标准化坐标系
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    # 不透明度上均匀采样代替深度值
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    # 360度场景
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    # 每N个图像采用1个图像进行测试，默认N=8
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    # 训练数据的输出频率
    parser.add_argument("--i_print", type=int, default=100,
                        help='frequency of console printout and metric loggin')
    # tensorboard图像记录频率
    parser.add_argument("--i_img", type=int, default=500,
                        help='frequency of tensorboard image logging')
    # 权重保存频率
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    # 测试结果保存频率
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    # render_poses视频保存频率
    parser.add_argument("--i_video", type=int, default=50000,
                        help='frequency of render_poses video saving')

    return parser


def train():
    # 导入参数设置
    parser = config_parser()
    args = parser.parse_args()

    # Load data
    K = None
    if args.dataset_type == 'llff':
        # 利用load_llff_data()导入数据
        images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                  recenter=True, bd_factor=.75,
                                                                  spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        # 用来测试的数据id
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]  # 生成以llffhold为间隔的不超过图片数量的测试集id[0,8,16]

        # 验证集和测试集相同
        i_val = i_test
        i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                            (i not in i_test and i not in i_val)])  # 剩下部分当做训练集，
        # array([ 1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13, 14, 15, 17, 18, 19])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    # # 不同数据集类型不同处理方法
    # elif args.dataset_type == 'blender':
    #     images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    #     print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split
    #
    #     near = 2.
    #     far = 6.
    #
    #     if args.white_bkgd:
    #         images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    #     else:
    #         images = images[..., :3]
    #
    # elif args.dataset_type == 'LINEMOD':
    #     images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res,
    #                                                                                 args.testskip)
    #     print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
    #     print(f'[CHECK HERE] near: {near}, far: {far}.')
    #     i_train, i_val, i_test = i_split
    #
    #     if args.white_bkgd:
    #         images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])
    #     else:
    #         images = images[..., :3]
    #
    # elif args.dataset_type == 'deepvoxels':
    #
    #     images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
    #                                                              basedir=args.datadir,
    #                                                              testskip=args.testskip)
    #
    #     print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    #     i_train, i_val, i_test = i_split
    #
    #     hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
    #     near = hemi_R - 1.
    #     far = hemi_R + 1.
    #
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # 将内参转换为内参矩阵
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[i_test])

    # 创建log路径，保存训练用的所有参数到args，复制config参数并保存
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # create_nerf()加载网络模型
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)  # dict类型一开始有9个元素，update之后变为11个；
    render_kwargs_test.update(bds_dict)

    # 转到GPU
    render_poses = torch.Tensor(render_poses).to(device)

    # 如果只渲染
    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # 使用测试集
                images = images[i_test]
            else:
                # 使用render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname,
                                       'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                                  savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    N_rand = args.N_rand
    use_batching = not args.no_batching
    if use_batching:
        # 批量处理
        print('get rays')
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
        print('done, concats')
        rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
        rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
        rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)  # 打乱

        print('done')
        i_batch = 0

    # 转到GPU
    if use_batching:
        images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb).to(device)

    N_iters = 200000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)


    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        if use_batching:
            # 每次从所有ray中抽取N_rand个ray，每遍历一边就打乱顺序(shuffle)
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)  # [3,B,3]
            batch_rays, target_s = batch[:2], batch[2]  # batch_rays=torch.Size([2, 1024, 3]),target_s=torch.Size([1024, 3])

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # 重新洗牌
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        else:
            # 只从一张图片里面抽ray
            img_i = np.random.choice(i_train)  # 随机选择一张照片
            target = images[img_i]
            target = torch.Tensor(target).to(device)
            pose = poses[img_i, :3, :4]  # 获取该照片的target和poses

            if N_rand is not None:
                # 从当前图像随机选择N_rand个像素点采样
                # 射线参数
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                # 得到像素坐标
                if i < args.precrop_iters:  # 迭代次数小于预设值5时进行图像剪裁
                    dH = int(H // 2 * args.precrop_frac)
                    dW = int(W // 2 * args.precrop_frac)
                    coords = torch.stack(
                        torch.meshgrid(
                            torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                            torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                        ), -1)
                    if i == start:
                        print(
                            f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {args.precrop_iters}")
                else:
                    coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                         -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                # 随机选择一些像素点
                select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                # 随机选择的那些像素点的对应坐标
                select_coords = coords[select_inds].long()  # (N_rand, 2)
                # 对应的射线参数
                rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                # 对应的true value
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        # 体素渲染，得到output后即可开始梯度下降
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=i < 10, retraw=True,
                                        **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s) #  loss function
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # 学习率衰减
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0

        # 定期保存检查点
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # 定期生成训练过程中的渲染视频
        if i % args.i_video == 0 and i > 0:
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        # 定期生成测试集上的渲染图象
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
                            gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        # 定期打印loss
        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")

        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.FloatTensor')

    train()
