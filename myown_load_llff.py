# -*- coding: gbk -*-

import numpy as np
import os, imageio

# 规范化图片分辨率
# 两种类型的图片处理：factors传入下采样的参数、resolution传入规定大小的图像参数；
def _minify(basedir, factors=[], resolutions=[]):
    """

    :param basedir: 数据集存放的根目录
    :param factors: 压缩因子
    :param resolutions: 分辨率
    :return: 无返回值，仅用作图片处理
    """
    needtoload = False
    # 创建下采样的文件夹
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    # 如果两者都没有或者处理后的图片文件已存在，则不进行处理
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output
    # 遍历指定文件夹中的图片文件，并筛选出符合条件的图片（以JPG、jpg、png、jpeg、PNG结尾的文件）。
    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    # 对于给定的压缩因子（factors）和分辨率（resolutions），分别创建对应的子文件夹，并在其中进行图片的压缩和格式转换。
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        # 使用ImageMagick库中的mogrify命令对图片进行压缩和格式转换。
        # 对于每个子文件夹，首先将原始图片文件复制到该子文件夹中，然后使用mogrify命令将图片大小调整为指定的尺寸，并将格式转换为PNG格式。
        # 获取原始图片格式并构建mogrify命令参数
        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        # 进入子文件夹，并执行mogrify命令
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        # 如果原始图片的格式不是PNG，则删除原始图片文件，保留压缩后的PNG格式图片。
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    """
    加载数据，包括相机姿态、边界信息和图像数据。
    :param basedir: 数据集根目录。
    :param factor: 压缩因子
    :param width: W
    :param height: H
    :param load_imgs: 是否加载图像数据
    :return: 相机姿态数据，边界信息数据，图像数据
    """
    # 加载相机姿态和边界信息
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    '''
    .npy文件是一个shape为（N，17），dtype为float64的array，N代表数据集的个数（一共有N张图片），17代表位姿参数。
    poses_arr[:, :-2]代表取前15列，为一个（N,15）的array，
    reshape([-1, 3, 5])代表将（N,15）的array转换为（N,3,5）的array，也就是把15列的一维数据变为3*5的二维数据。
    transpose([1,2,0])是将array的坐标系调换顺序，0换到2, 1、2换到0、1，shape变为（3,5,N）;
    最后poses输出的是一个（3,5,N）的array
    '''

    bds = poses_arr[:, -2:].transpose([1, 0])
    '''
    poses_arr[:, -2:].transpose([1,0])则是先提取poses_arr的后两列数据（N，2），然后将0,1坐标系对调，得到（2,N）shape的array：bds
    bds指的是bounds深度范围(near, far)
    '''

    # img0是N张图像中的第一张图像的路径名称
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape  # 读取图片大小为(H, W, 3)

    sfx = ''

    # 判断是否有下采样的相关参数，如果有，则对图像进行下采样
    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    # 判断是否存在下采样的路径
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    # 判断pose数量与图像个数是否一致，
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print('Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]))
        return

    # 获取处理后的图像shape，sh=(H/F,W/F,3)
    # 3*5矩阵由旋转矩阵，平移向量和hwf组成
    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor


    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    # 读取所有图像数据并把值缩小到0-1之间，imgs存储所有图片信息，大小为(H/F,W/F,3,N)
    imgs = imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)

    print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    # 用y轴和z轴叉乘得到x轴，再用x轴和z轴叉乘重新确立y轴方向确保垂直(初始的y轴方向up是估计得到的)
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

# 坐标系的转换(相机坐标系->世界坐标系)
def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)  # 对多个相机的中心进行求均值得到center
    vec2 = normalize(poses[:, :3, 2].sum(0))  # 对所有相机的Z轴求平均得到vec2向量
    up = poses[:, :3, 1].sum(0)  # 对所有的相机的Y轴求平均得到up向量
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)  # 输入给viewmatrix即可得到平均相机位姿

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    """
    生成螺旋路径上的渲染姿势
    :param c2w: 旋转矩阵
    :param up: y
    :param rads: 螺旋路径的半径
    :param focal: 焦距
    :param zdelta: z轴增量
    :param zrate: z轴速率
    :param rots: 旋转角度
    :param N: 生成的姿势数量
    :return: 旋转路径上的渲染姿势列表
    """
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        # 每一迭代生成一个新的相机位置。
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    # 把相机位姿转换成与世界坐标一致
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)  # 得到平均位姿
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses  # 左乘c2w的逆，完成位姿旋转
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


# 球面相机分布
def spherify_poses(poses, bds):
    """球面化相机分布
    """
    # 将 3x4 的姿势数组扩展为 4x4
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    # 找到离所有相机中心射线距离之和最短的点(约等于场景的中心位置)
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    # 把pt_mindist挪到世界坐标系原点，并将所有相机z轴的平均方向与世界坐标系的z轴同向
    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    # 把相机位置缩放到单位圆内
    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    # 计算球形坐标系参数
    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]  # 球心
    radcircle = np.sqrt(rad ** 2 - zh ** 2)  # 半径

    # 生成新姿势
    new_poses = []
    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    # 合并新姿势和初始姿势
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    """
    加载llff型数据集
    :param basedir: 基础目录
    :param factor: 下采样因子
    :param recenter: 是否重新调整相机姿势
    :param bd_factor: 边界框的缩放因子
    :param spherify: 是否对相机姿势进行球化
    :param path_zflat: 是否将相机路径沿z轴拉平
    :return: 包含图像、姿势、边界框、渲染姿势和留出视图索引的元组
    """
    poses, bds, imgs = _load_data(basedir, factor=factor)  # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())

    # 把DRB转换成RUB格式
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # 如果提供了bd_factor，则进行缩放
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    # 如果recenter为True，则重新调整姿势
    if recenter:
        poses = recenter_poses(poses)

    # 如果启用spherify，则对姿势进行球化
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 平均姿势并生成螺旋路径
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3, :4])

        up = normalize(poses[:, :3, 1].sum(0))

        # 计算一个比较合理的焦距
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # 获取用于螺旋路径的半径
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # 相机的位置坐标
        rads = np.percentile(np.abs(tt), 90, 0)  # 估算相机位置在三维空间中的范围，可以被近似当作半径
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:  # 沿z轴拉平
            zloc = -close_depth * .1  # 相机在 z 轴上的位置
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # 生成spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test



