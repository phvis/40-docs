# 项目概述

​		本文重点介绍如何利用飞桨生成对抗网络`PaddleGAN` 在风格迁移数据集上，使用当前`PaddleGAN`的`Pix2pix`模型完风格迁移任务。通过Pix2pix模型实现图像翻译，从而应用到图像生成等领域。

​		**关键词: 图像超分、Pix2pix、PaddleGAN**

## 文档目录结构

- (1) 模型简述
- (2) 环境安装
  - (2.1) `PaddlePaddle`安装
    - (2.1.1) 安装对应版本`PaddlePaddle`
    - (2.1.2) 验证安装是否成功
  - (2.2) `PaddleGAN`安装
    - (2.2.1) 下载`PaddleGAN`代码
    - (2.2.2) 安装依赖项目
    - (2.2.3) 验证安装是否成功
- (3) 数据准备
  - (3.1) facade数据集
  - (3.2) 自制数据集
- (4) 模型训练
  - (4.1) 训练前数据准备
  - (4.2) 开始训练
  - (4.3) 可视化训练
  - (4.4) 回复训练
  - (4.5) 多卡训练
- (5) 模型验证与预测
  - (5.1) 开始验证
  - (5.2) 开始预测
- (6) 模型部署与转化
- (7) 配置文件的说明
  - (7.1) 整体配置文件格式综述
  - (7.2) 数据路径与数据预处理说明
  - (7.3) 模型与损失函数说明
  - (7.4) 优化器说明
  - (7.5) 其它参数说明
- (8) 部分参数值推荐说明
  - (8.1) 训练批大小
  - (8.2) 训练轮次大小
  - (8.3) 训练学习率大小
  - (8.4) 配置文件说明

# (1) 模型简述

​        Pix2pix利用成对的图片进行图像翻译，即输入为同一张图片的两种不同风格，可用于进行风格迁移。Pix2pix是在cGAN的基础上进行改进的，cGAN的生成网络不仅会输入一个噪声图片，同时还会输入一个条件作为监督信息，pix2pix则是把另外一种风格的图像作为监督信息输入生成网络中，这样生成的fake图像就会和作为监督信息的另一种风格的图像相关，从而实现了图像翻译的过程。


![pix2pix](J:\model_doc\GAN\Pix2Pix\docs.assets\pix2pix.png)

# (2) 环境安装

## (2.1) `PaddlePaddle`安装

### (2.1.1) 安装对应版本`PaddlePaddle`

​		根据系统和设备的`cuda`环境，选择对应的安装包，这里默认使用`pip`在`linux`设备上进行安装。

![176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9](docs.assets/176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9.png)

​		在终端中执行:

```bash
pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

​		安装效果:

![image-20221127141851237](docs.assets/image-20221127141851237.png)

### (2.1.2) 验证安装是否成功

```bash
# 安装完成后您可以使用 python进入python解释器，
python
# 继续输入
import paddle 
# 再输入 
paddle.utils.run_check()
```

​		如果出现`PaddlePaddle is installed successfully!`，说明您已成功安装。

![image-20221127142019376](docs.assets/image-20221127142019376.png)

## (2.2) `PaddleGAN`安装

### (2.2.1) 下载`PaddleGAN`代码

​		用户可以通过使用`github`或者`gitee`的方式进行下载，我们当前版本为`PaddleGAN`的release v2.5版本。后续在使用时，需要对应版本进行下载。

![20221211021030.png](docs.assets/20221211021030.png)

```bash
# github下载
git clone -b release/2.5 https://github.com/PaddlePaddle/PaddleGAN.git
# gitee下载
git clone -b release/2.5 https://gitee.com/PaddlePaddle/PaddleGAN.git
```

### (2.2.2) 安装依赖项目

* 方式一：
  通过直接`pip install` 安装，可以最高效率的安装依赖

``` bash
pip install --upgrade ppgan
```

* 方式二：
  下载`PaddleGAN`代码后，进入`PaddleGAN`代码文件夹目录下面

``` bash
cd PaddleGAN
pip install -v -e .  # or "python setup.py develop"

# 安装其他依赖
pip install -r requirements.txt
```

### (2.2.3) 其他第三方工具安装

* 涉及视频的任务都需安装**ffmpeg**，这里推荐使用[conda](https://docs.conda.io/en/latest/miniconda.html)安装：

```
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
```

* 如需使用可视化工具监控训练过程，请安装[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)：

```
python -m pip install visualdl -i https://mirror.baidu.com/pypi/simple
```

*注意：VisualDL目前只维护Python3以上的安装版本

# (3) 数据准备

pix2pix所使用的facades数据的组成形式为：

	facades
	 ├── test
	 ├── train
	 └── val

## (3.1)facade数据集

- 从网页下载

pixel2pixel模型相关的数据集可以在[这里](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/)下载，下载后记得软连接到 ```PaddleGAN/data/``` 下。

- #### 使用脚本下载


我们在 ```PaddleGAN/data``` 文件夹下提供了一个脚本 ```download_pix2pix_data.py``` 方便下载pix2pix模型相关的数据集。

目前支持下载的数据集名称有：apple2orange, summer2winter_yosemite,horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos, cityscapes。

同理，执行如下命令，可以下载对应的数据集到 ```~/.cache/ppgan``` 并软连接到 ```PaddleGAN/data/``` 下。

```
python data/download_pix2pix_data.py --name cityscapes
```
 - wget下载

```bash
  wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz --no-check-certificate
```



## (3.2) 自制数据集

### 成对数据集构建

针对需要成对数据训练的模型，如Pixel2Pixel等，如需使用自己的数据集，需要构造成如下目录的格式。

注意图片应该制作成下图的样式，即左边为一种风格，另一边为相应转换的风格。

```
facades
├── test
├── train
└── val
```

![](docs.assets\1.jpg)

![20221210132133889](docs.assets\20221210132133889.jpg)


# (4) 模型训练

## (4.1) 训练前准备

 **修改选中模型的配置文件**

 所有模型的配置文件均在``` PaddleGAN/configs ```目录下。 找到你需要的模型的配置文件，修改模型参数，一般修改迭代次数，num_workers，batch_size以及数据集路径。

 找到``` /home/aistudio/PaddleGAN/configs ```目录，修改配置文件``pix2pix_facades.yaml``中的

-  参数``epochs``设置为 200

-  参数``dataset：train：num_workers``设置为4

-  参数``dataset：train：batch_size``设置为1

-  参数``dataset：train：dataroot``改为data/facades/train

-  参数``dataset：test：dataroot``改为data/facades/test

-  configs/keypoint/tiny_pose/tinypose_128x96.yml


```yaml
dataset:
  train:
    name: PairedDataset
    dataroot: data/facades/train
    num_workers: 4
    batch_size: 1
    preprocess:
      - name: LoadImageFromFile
        key: pair
      - name: SplitPairedImage
        key: pair
        paired_keys: [A, B]
      - name: Transforms
        input_keys: [A, B]
        pipeline:
          - name: Resize
            size: [286, 286]
            interpolation: 'bicubic' #cv2.INTER_CUBIC
            keys: [image, image]
          - name: PairedRandomCrop
            size: [256, 256]
            keys: [image, image]
          - name: PairedRandomHorizontalFlip
            prob: 0.5
            keys: [image, image]
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [127.5, 127.5, 127.5]
            std: [127.5, 127.5, 127.5]
            keys: [image, image]
  test:
    name: PairedDataset
    dataroot: data/facades/test
    num_workers: 4
    batch_size: 1
    preprocess:
      - name: LoadImageFromFile
        key: pair
    - name: Transforms
        input_keys: [A, B]
        pipeline:
          - name: Resize
            size: [256, 256]
            interpolation: 'bicubic' #cv2.INTER_CUBIC
            keys: [image, image]
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [127.5, 127.5, 127.5]
            std: [127.5, 127.5, 127.5]
            keys: [image, image]

```

## (4.2) 开始训练

​		请确保已经完成了`PaddleGAN`的安装工作，并且当前位于`PaddleGAN`目录下，执行以下脚本：

```bash
%cd /home/aistudio/PaddleGAN/
!export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡

!python -u tools/main.py --config-file configs/pix2pix_facades.yaml
```

​	执行效果:

![20221211131251911](J:\model_doc\GAN\Pix2Pix\docs.assets\20221211131251911.jpg)



## (4.3) 可视化训练

[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)是针对深度学习模型开发所打造的可视化分析工具，提供关键指标的实时趋势可视化、样本训练中间过程可视化、网络结构可视化等等，更能直观展示超参与模型效果间关系，辅助实现高效调参。

以下操作请确保您已完成[VisualDL](https://github.com/PaddlePaddle/VisualDL)的安装，安装指南请见[VisualDL安装文档](https://github.com/PaddlePaddle/VisualDL/blob/develop/README_CN.md#%E5%AE%89%E8%A3%85%E6%96%B9%E5%BC%8F)。

**通过在配置文件 pix2pix_facades.yaml 中添加参数`enable_visualdl: true`使用 [飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)对训练过程产生的指标或生成的图像进行记录，并运行相应命令对训练过程进行实时监控：**

![119621184-68c32e80-be38-11eb-9830-95429db787cf](J:\model_doc\GAN\Pix2pix\docs.assets\119621184-68c32e80-be38-11eb-9830-95429db787cf.png)

如果想要自定义[飞桨VisualDL](https://github.com/PaddlePaddle/VisualDL)可视化内容，可以到 [./PaddleGAN/ppgan/engine/trainer.py](https://github.com/PaddlePaddle/PaddleGAN/blob/develop/ppgan/engine/trainer.py) 中进行修改。

本地启动命令：

```
visualdl --logdir output_dir/pix2pix_facade-2022-11-29-09-21/
```

更多启动方式及可视化功能使用指南请见[VisualDL使用指南](https://github.com/PaddlePaddle/VisualDL/blob/develop/docs/components/README_CN.md)。

## (4.4) 恢复训练

​	    在训练过程中默认会**保存上一个epoch的checkpoint在`output_dir`中，方便恢复训练。**

​	    本次示例中，cyclegan的训练默认**每五个epoch会保存checkpoint**，如需更改，可以到**config文件中的`interval`**进行修改。

![120147997-2758c780-c21a-11eb-9cf1-4288dbc01d22](J:\model_doc\GAN\Pix2pix\docs.assets\120147997-2758c780-c21a-11eb-9cf1-4288dbc01d22.png)

```bash
python -u tools/main.py --config-file configs/pix2pix_facades.yaml --resume your_checkpoint_path
```

​	`--resume (str)`: 用来恢复训练的checkpoint路径（保存于上面配置文件中设置的output所在路径）。



## (4.5) 多卡训练

```bash
!CUDA_VISIBLE_DEVICES=0,1,2,3
!python -m paddle.distributed.launch tools/main.py --config-file configs/pix2pix_facades.yaml
```

​		执行效果:

![image-20221127183832898](docs.assets/20221210133010717.jpg)

# (5) 模型验证与预测

## (5.1) 开始验证

​		训练完成后，用户可以使用评估脚本`tools/eval.py`来评估模型效果。运行``/home/aistudio/pretrained_model/Pix2pix_PSNR_50000_weight.pdparams``代码测试 Pix2pix 模型。

```bash
%cd /home/aistudio/PaddleGAN/

!python tools/main.py --config-file configs/pix2pix_facades.yaml \
    --evaluate-only --load /home/aistudio/pretrained_model/pix2pix_facades_50000_weight.pdparams
```

​		执行效果:

![20221211131804253](J:\model_doc\GAN\Pix2Pix\docs.assets\20221211131804253.jpg)

​	

## (5.4) 开始预测

​		除了可以分析模型的准确率指标之外，我们还可以对一些具体样本的预测。

```bash
%cd /home/aistudio/PaddleGAN/

!python tools/main.py --config-file configs/pix2pix_facades.yaml \
    --evaluate-only --load /home/aistudio/pretrained_model/pix2pix_facades_50000_weight.pdparams
```

​		执行效果:

![A2B](docs.assets\A2B.png)


- `--evaluate-only`: 是否仅进行预测。

- `--load (str)`: 训练好的权重路径。

  

# (6) 模型部署与转化

```python
# 导出行人检测模型
!python tools/export_model.py -c configs/picodet/application/pedestrian_detection/picodet_s_320_lcnet_pedestrian.yml \
        -o weights=https://paddledet.bj.bcebos.com/models/picodet_s_320_lcnet_pedestrian.pdparams

```



# (7) 配置文件说明

​		正是因为有配置文件的存在，我们才可以使用更便捷的进行消融实验。在本章节中我们选择
```PaddleGAN/configs/pix2pix_facades.yaml```文件来进行配置文件的详细解读。

## (7.1) 整体配置文件格式综述

我们将```pix2pix_facades.yml```进行拆分解释

* **Pix2pix** 表示模型的名称 *Pix2pix*
* **facades** 表示数据集为街景分隔数据集

一个模型的配置文件按功能可以分为:

- **主配置文件入口**: `pix2pix_facades.yml`

## (7.2) 数据路径与数据预处理说明

​		这一小节主要是说明数据部分，当准备好数据，如何进行配置文件修改，以及该部分的配置文件有什么内容。

**首先是进行数据路径配置

```yaml
ataset:
  train:
    name: PairedDataset
    dataroot: data/facades/train
    num_workers: 4
    batch_size: 1
    preprocess:
      - name: LoadImageFromFile
        key: pair
      - name: SplitPairedImage
        key: pair
        paired_keys: [A, B]
      - name: Transforms
        input_keys: [A, B]
        pipeline:
          - name: Resize
            size: [286, 286]
            interpolation: 'bicubic' #cv2.INTER_CUBIC
            keys: [image, image]
          - name: PairedRandomCrop
            size: [256, 256]
            keys: [image, image]
          - name: PairedRandomHorizontalFlip
            prob: 0.5
            keys: [image, image]
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [127.5, 127.5, 127.5]
            std: [127.5, 127.5, 127.5]
            keys: [image, image]
  test:
    name: PairedDataset
    dataroot: data/facades/test
    num_workers: 4
    batch_size: 1
    preprocess:
      - name: LoadImageFromFile
        key: pair
      - name: SplitPairedImage
        key: pair
        paired_keys: [A, B]
      - name: Transforms
        input_keys: [A, B]
        pipeline:
          - name: Resize
            size: [256, 256]
            interpolation: 'bicubic' #cv2.INTER_CUBIC
            keys: [image, image]
          - name: Transpose
            keys: [image, image]
          - name: Normalize
            mean: [127.5, 127.5, 127.5]
            std: [127.5, 127.5, 127.5]
            keys: [image, image]
```

## (7.3) 模型与损失函数说明

当我们配置好数据后，下面在看关于模型和主干网络的选择

``` yaml
epochs: 200
output_dir: output_dir

model:
  name: Pix2PixModel
  generator:
    name: UnetGenerator
    norm_type: batch
    input_nc: 3
    output_nc: 3
    num_downs: 8 #unet256
    ngf: 64
    use_dropout: False
  discriminator:
    name: NLayerDiscriminator
    ndf: 64
    n_layers: 3
    input_nc: 6
    norm_type: batch
  direction: b2a
  pixel_criterion:
    name: L1Loss
    loss_weight: 100
  gan_criterion:
    name: GANLoss
    gan_mode: vanilla
```

  **Note**

* 我们模型的`architecture`是`PicoDet`。
* 主干网络是 `LCNet`，在这里我们可以自由更换，比如换成`ResNet50_vd`, 不同的主干网络需要选择不同的参数。
* `nms` 此部分内容是预测与评估的后处理，一般可以根据需要调节`threshold`参数来优化处理效果。

## (7.4) 优化器说明

当我们配置好数据与模型后，下面再看关于优化器的选择

``` yaml
# 优化器配置
lr_scheduler:
  name: LinearDecay
  learning_rate: 0.0002
  start_epoch: 100
  decay_epochs: 100
  # will get from real dataset
  iters_per_epoch: 1
```

## (7.5) 其它参数说明

``` yaml
optimizer:
  optimG:
    name: Adam
    net_names:
      - netG
    beta1: 0.5
  optimD:
    name: Adam
    net_names:
      - netD
    beta1: 0.5

log_config:
  interval: 100
  visiual_interval: 500

snapshot_config:
  interval: 5

validate:
  interval: 4000
  save_img: false
  metrics:
    fid: # metric name, can be arbitrary
        name: FID
        batch_size: 8

export_model:
  - {name: 'netG', inputs_num: 1}
```

# (8) 部分参数值推荐说明

## (8.1) 训练批大小

```yaml
batch_size: 64
```

​		批大小(batch_size)通常取值: **32, 64, 128, 256, 512**。

​		一般可以按照数据集中训练的样本(图像)数量大小以及期望一轮训练迭代次数来大致取值。

- 如果数据集训练样本数量为: `N`
- 期望一轮训练迭代次数为: `I`
- 得到大致`batch_size`大小: `B = N/I`

如果B大于32小于64，则可以选32；以此类推。

**Note**

- `batch_size`会收显存大小影响，因此过大的批大小可能大致运行训练失败——因为GPU显存不够。
- `batch_size` 是训练神经网络中的一个重要的超参数，该值决定了一次将多少数据送入神经网络参与训练。论文 [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)，当 `batch size` 的值与学习率的值呈线性关系时，收敛精度几乎不受影响。在训练 ImageNet 数据时，大部分的神经网络选择的初始学习率为 0.1，`batch size` 是 256，所以根据实际的模型大小和显存情况，可以将学习率设置为 0.1*k, batch_size 设置为 256*k。在实际任务中，也可以将该设置作为初始参数，进一步调节学习率参数并获得更优的性能。

## (8.2) 训练轮次大小

```bash
total_iters: 1000000
```

​		总轮次(`total_iters`)通常取值: **1000000。**

​		如果取1000000轮效果不理想，可以用10000000轮尝试，如果效果有提升则可以用大的训练轮次进行训练。

## (8.3) 训练学习率大小

```yaml
learning_rate: 0.0002
```

​		学习率(`learning_rate`)通常取配置文件的默认值，如果性能不好，可以尝试调小或调大，公式: $new\_lr=lr * ratio$。其中调小时: `ratio`可以取`0.5`或者`0.1`；而调大时:  `ratio`可以取或`1.0`者`2.0`。但学习率一般不超过1.0，否则容易训练不稳定。

​		如果配置文件所对应的模型默认为N卡训练的模型，则需要对学习率除以卡数N: $new\_lr=lr / N$。

​		由于本模型默认为4卡训练的，因此如果是在单卡上训练该模型需要修改学习率为`0.08`。

## (8.4) 配置文件说明



###  (8.4.1)Config文件参数介绍

以`lapstyle_rev_first.yaml`为例。

 Global

| 字段                      | 用途                           | 默认值       |
| ------------------------- | ------------------------------ | ------------ |
| total_iters               | 设置总训练步数                 | 30000        |
| min_max                   | tensor数值范围（存图像时使用） | (0., 1.)     |
| output_dir                | 设置输出结果所在的文件路径     | ./output_dir |
| snapshot_config: interval | 设置保存模型参数的间隔         | 5000         |

###  (8.4.2)Model

| 字段                    | 用途                         | 默认值                              |
| :---------------------- | ---------------------------- | ----------------------------------- |
| name                    | 模型名称                     | LapStyleRevFirstModel               |
| revnet_generator        | 设置revnet生成器             | RevisionNet                         |
| revnet_discriminator    | 设置revnet判别器             | LapStyleDiscriminator               |
| draftnet_encode         | 设置draftnet编码器           | Encoder                             |
| draftnet_decode         | 设置draftnet解码器           | DecoderNet                          |
| calc_style_emd_loss     | 设置style损失1               | CalcStyleEmdLoss                    |
| calc_content_relt_loss  | 设置content损失1             | CalcContentReltLoss                 |
| calc_content_loss       | 设置content损失2             | CalcContentLoss                     |
| calc_style_loss         | 设置style损失2               | CalcStyleLoss                       |
| gan_criterion: name     | 设置GAN损失                  | GANLoss                             |
| gan_criterion: gan_mode | 设置GAN损失模态参数          | vanilla                             |
| content_layers          | 设置计算content损失2的网络层 | ['r11', 'r21', 'r31', 'r41', 'r51'] |
| style_layers            | 设置计算style损失2的网络层   | ['r11', 'r21', 'r31', 'r41', 'r51'] |
| content_weight          | 设置content总损失权重        | 1.0                                 |
| style_weigh             | 设置style总损失权重          | 3.0                                 |

###  (8.4.3)Dataset (train & test)

| 字段         | 用途                             | 默认值               |
| :----------- | -------------------------------- | -------------------- |
| name         | 数据集名称                       | LapStyleDataset      |
| content_root | 数据集所在路径                   | data/coco/train2017/ |
| style_root   | 目标风格图片所在路径             | data/starrynew.png   |
| load_size    | 输入图像resize后图像大小         | 280                  |
| crop_size    | 随机剪裁图像后图像大小           | 256                  |
| num_workers  | 设置工作进程个数                 | 16                   |
| batch_size   | 设置一次训练所抓取的数据样本数量 | 5                    |

###  (8.4.4)Lr_scheduler 

| 字段          | 用途             | 默认值         |
| :------------ | ---------------- | -------------- |
| name          | 学习策略名称     | NonLinearDecay |
| learning_rate | 设置初始学习率   | 1e-4           |
| lr_decay      | 设置学习率衰减率 | 5e-5           |

###  (8.4.5)Optimizer

| 字段      | 用途                | 默认值  |
| :-------- | ------------------- | ------- |
| name      | 优化器类名          | Adam    |
| net_names | 优化器作用的网络    | net_rev |
| beta1     | 设置优化器参数beta1 | 0.9     |
| beta2     | 设置优化器参数beta2 | 0.999   |

###  (8.4.6) Validate

| 字段     | 用途               | 默认值 |
| :------- | ------------------ | ------ |
| interval | 设置验证间隔       | 500    |
| save_img | 验证时是否保存图像 | false  |

###  (8.4.7) Log_config

| 字段             | 用途                             | 默认值 |
| :--------------- | -------------------------------- | ------ |
| interval         | 设置打印log间隔                  | 10     |
| visiual_interval | 设置训练过程中保存生成图像的间隔 | 500    |
