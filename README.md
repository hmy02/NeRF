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

实际上，文章最大体的方向是很好理解的，就是利用有限个拍摄角度得到的图像，通过深度学习来拟合出任意拍摄角度的图像。听起来好像很简单，但事实上如果想仅仅使用个位数维度的拍摄角度来拟合一个成十万百万维的图像，显然是不现实的。文章采用了立体空间中相机到像素的射线上的点来拟合该像素的rgb值的方法，有效的提升了输入维，减少了输出维，将问题拆解成了预测单个像素点。那么具体是怎么实现的呢，下面来拆解一下NeRF的训练框架。

### step1 数据集获取

作为练习目的，download_example_data.sh文件中提供了fern数据集以及lego数据集的下载路径，更多的数据集可以在[link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)里下载。也可以自发的通过相机拍摄一组照片，并通过COLMAP制作属于自己的数据集(COLMAP可以根据你给出的照片集估计出训练必须的poses_bounds.npy)

### step2 数据加载

定义了一个load_llff_data()方法，首先将给定的图片按照焦距进行放缩，同时将poses_bounds.npy中每张图片对应的17个参数分别转换成位姿矩阵和深度范围。之后根据已有相机位姿的平均值确立一个世界中心，并把所有的相机位姿按世界坐标重新标记。然后选择使用球面化或者螺旋化位姿，生成一个位姿序列，这些序列覆盖角度比原始序列全，可以被用作模型测试。

### step3 训练网络

首先这里需要先明确网络结构，网络的输入是由空间内某像素到相机位置确立射线的上的点以及射线的方向组成(也可以通过 $\gamma(p)$函数进行增维)，首先输入点的坐标，通过若干层网络得到一个维度为W的输出，并将这层的输出分开输入到一个 $W\cdot 1$ 的线性层和 $W\cdot W$ 的线性层中，前者的输出作为不透明度记录，后者送入下一个若干层网络并输出一个三维的rgb值，整个网络一共输出四个数。接下来要根据射线上的这些点的rgb值和不透明度的加权求和，得到该像素点的值，这个过程被称为体素渲染。这样我们完整的从数据到像素值的过程就构建出来了，通过对比真实像素值以及拟合的像素值，对整个过程做梯度下降，即可得到一个不错的模型。

具体实施：根据某一像素的位置和相机位置确立一条射线，这条射线上先随机取 $N_c$ 个点，先通过这些粗样本送进网络得到输出，并进行体素渲染估计出权重(基于不透明度)后可以得到一条密度曲线，再根据密度曲线均匀或随机的取更多的点( $N_f$ 个)送入网络重复该过程，新取得的点被称为精细样本。

### step 4 测试阶段

现在我们已经拥有了一个很好的模型，在测试阶段，固定某一拍摄点位，通过输入相机位姿，图像长宽，最近端最远端等参数，进而生成相机到全部像素点的射线，并取点送入网络，通过体素渲染预测该像素点rgb值，即可利用全部像素点拼成一张该拍摄点位'拍摄'得到的照片。

## 一些PhotometricStereo的工作中idea的粗略汇总

### Scalable, Detailed, and Mask-Free Universal Photometric Stereo*

### SR-PSN: Estimating High-resolution Surface Normals via Low-resolution Photometric Stereo Images*

### GPS-Net: Graph-based Photometric Stereo Network

逐像素法: 逐像素法在处理光度立体时，将每个像素观察到的值投影到一个固定大小的观测图中，然后探索图像之间的强度变化。然而，这种方法的缺点在于，观测图的大小需要合适地设置。如果设置得太大，有效数据只占观测图的一部分，导致信息浪费；而设置得太小，则观测图的分辨率会降低。这种方法忽略了图像内部的空间域特征，并且受到了观测图大小的限制，要么选择分辨率，要么选择输入图像的密度，导致当输入图像数量增加时性能下降。

全像素法: 全像素法则通过将输入的光度图通过特征提取器得到每个图像的特征信息，然后通过最大池化层融合这些特征图，从而获得图像内部的强度变化。这种方法对每个输入图像都产生一个特征图，保留了原始图像的空间域信息。然而，它也存在一些缺点，比如无法兼顾多幅图像之间的光照变化，而且使用过多的卷积层会导致梯度消失，从而降低了最终结果的质量。

文章说结合了逐像素法和全像素法，规避了双方的缺点，感觉就是先用GNN来了个逐像素法保留空间特征，再给这些输出按原来的图片拼回去，再来一CNN，就结束了。

### Leveraging Spatial and Photometric Context for Calibrated Non-Lambertian Photometric Stereo

不同光照方向上的某个点会呈现出不同的像素值，将这些像素值按照光照方向拼接起来得到一个像素矩阵，选取物体上小范围内若干个这样的像素矩阵组成的集合作为输入，通过CNN拟合一个热图，热图最热的地方作垂线交于半球体，交点与这个小范围的中心相连即可得到这个小范围的表面法线方向。

### DiLiGenT-Π: Photometric Stereo for Planar Surfaces with Rich Details – Benchmark Dataset and Beyond

提供了一个专注近平面细节的数据集，可以应用于模型微调，使模型恢复表面细节的能力得到提升












