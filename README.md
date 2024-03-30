# NeRF-pytorch
## How to run

To train a low-res `fern` NeRF:
```
cd NeRF
python run_nerf.py --config configs/fern.txt
```
After training for 200k iterations (~8 hours on a single 2080 Ti), you can find the following video at `logs/fern_test/fern_test_spiral_200000_rgb.mp4` and `logs/fern_test/fern_test_spiral_200000_disp.mp4`

![](https://user-images.githubusercontent.com/7057863/78473081-58ea1600-7770-11ea-92ce-2bbf6a3f9add.gif)

---

## 训练框架

### step1 数据集获取

作为练习目的，download_example_data.sh文件中提供了fern数据集以及lego数据集的下载路径，更多的数据集可以在[link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)里下载。也可以自发的通过相机拍摄一组照片，并通过COLMAP制作属于自己的数据集(COLMAP可以根据你给出的照片集估计出训练必须的poses_bounds.npy)

### step2 load_llff_data()

load_llff_data()首先将给定的图片按照焦距进行放缩，同时将poses_bounds.npy中每张图片对应的17个参数分别转换成位姿矩阵和深度范围。之后根据已有相机位姿的平均值确立一个世界中心，并把所有的相机位姿按世界坐标重新标记。然后选择使用球面化或者螺旋化位姿，生成一个位姿序列，这些序列覆盖角度比原始序列全，可以被用作模型测试。

### step3 训练网络

首先这里需要先明确网络结构，网络的输入是由空间内某像素到相机位置确立射线的上的点以及射线的方向组成(也可以通过 $\gamma(p)$函数进行增维)，首先输入点的坐标，通过若干层网络得到一个维度为W的输出，并将这层的输出分开输入到一个W*1的线性层和W*W的线性层中，前者的输出作为不透明度记录，后者送入下一个若干层网络并输出一个三维的rgb值，整个网络一共输出四个数。接下来要根据射线上的这些点的rgb值和不透明度的加权求和，得到该像素点的值，这个过程被称为体素渲染。这样我们完整的从数据到像素值的过程就构建出来了，通过对比真实像素值以及拟合的像素值，对整个过程做梯度下降，即可得到一个不错的模型。

具体实施：根据某一像素的位置和相机位置确立一条射线，这条射线上先随机取 $N_c$ 个点，先通过这些粗样本送进网络得到输出，并进行体素渲染估计出权重(基于不透明度)后可以得到一条密度曲线，再根据密度曲线均匀或随机的取更多的点( $N_f$ 个)送入网络重复该过程，新取得的点被称为精细样本。

### step 4

现在我们已经拥有了一个很好的模型，在测试阶段，固定某一拍摄点位，通过输入相机位姿，图像长宽，最近端最远端等参数，进而生成相机到全部像素点的射线，并取点送入网络，通过体素渲染预测该像素点rgb值，即可利用全部像素点拼成一张该拍摄点位'拍摄'得到的照片。

## 一些近期PhotometricStereo的工作中idea的粗略汇总

### Leveraging Spatial and Photometric Context for Calibrated Non-Lambertian Photometric Stereo







