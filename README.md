  <h1 align="left">Image2latex 项目</h1>




<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary><h2 style="display: inline-block">目录</h2></summary>
  <ol>
    <li>
      <a href="# 关于本项目">关于本项目</a>
      <ul>
        <li><a href="####模型">模型</a></li>
        <li><a href="# 环境依赖">环境依赖</a></li>
      </ul>
    </li>
    <li>
      <a href="#快速开始">快速开始</a>
      <ul>
        <li><a href="#先决条件">数据集下载</a></li>
        <li><a href="#安装">环境安装</a></li>
      </ul>
    </li>
    <li>
      <a href="##使用方法">使用方法</a>
      <ul>
        <li><a href="#训练">训练</a></li>
        <li><a href="#测试<">测试</a></li>
        <li><a href="#计算BLEU得分">计算BLEU得分</a></li>
      </ul>
     </li>
    <li><a href="#结果">结果</a></li>
    <li><a href="#pretrained-model">Pretrained model</a></li>
  </ol>
</details>






<!-- 关于本项目 -->

## 关于本项目
本项目是进行图片到latex的翻译任务


### 模型
在本项目中，使用的是典型的Encoder-Decoder模型

#### 模型1:
- Encoder: 带Global Context(GC) Block的ResNet-31：8,4 times downsampling
- Decoder: 3层Transformer Decoder

网络结构为:

![MASTER's architecture](./imgs/Master.png)


### 环境依赖

本项目主要需要如下的环境依赖，具体安装方式见快速开始！
* [torch-1.10.0+cu113](https://pytorch.org/get-started/locally/)
* [MMOCR-0.2.0](https://github.com/open-mmlab/mmocr/tree/v0.2.0)
* [MMDetection-2.11.0](https://github.com/open-mmlab/mmdetection/tree/v2.11.0)
* [mmcv-full-1.4.0](https://github.com/open-mmlab/mmcv/tree/v1.4.0)



<!-- 环境安装 -->

## 快速开始

### 数据集下载

+ coming soon
### 环境安装

1. 创建新环境
   ```sh
   conda create -n im2latex python=3.7
   conda activate im2latex
   # 安装nltk
   pip install nltk
   ```

2. 安装torch1.10.0+cu113
   ```sh
   # install torch1.10.0+cu113
   pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
   ```

3. 安装 mmcv-full-1.4.0。点击 [here](https://github.com/open-mmlab/mmcv) 查看更多细节。

   ```sh
   # install mmcv-full-1.4.0 with torch version 1.10.0 cuda_version 11.3
   pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
   ```
   如果遇到网络问题无法下载，可选择手动下载mmcv1.4.0[安装包](https://disk.pku.edu.cn:443/link/B08D90D89CA352CCBB99D40E9B0E7A0E)，选择合适的版本进行安装，示例安装命令：
   ```sh
   pip install mmcv_full-1.4.0-cp38-cp38-manylinux1_x86_64.whl
   ```
   
4. 确保在项目文件夹下(im2latex)，安装 mmdetection。点击 [here](https://github.com/open-mmlab/mmdetection/blob/v2.11.0/docs/get_started.md) 查看更多细节。
   
   ```sh
   # We embed mmdetection-2.11.0 source code into this project.
   # You can cd and install it (recommend).
   cd ./mmdetection-2.11.0
   pip install -v -e .
   ```
   
5. 安装mmocr. 点击[here](https://github.com/open-mmlab/mmocr/blob/main/docs/install.md) 查看更多细节。

   ```sh
   # install mmocr
   cd {Path to im2latex}
   pip install -v -e .
   ```






<!-- USAGE EXAMPLES -->

## 使用方法

### 训练

1. ResNet31withGCB + 3\*Transformer Decoder
   ```shell
   sh im2latex/im2latex_resnet31withGCB.sh
   ```

   训练过程中的日志文件和checkpoint将会保存在 [expr_result/im2latex_res31](expr_result/im2latex_res31) 中

### 预测

1. ResNet31withGCB + 3\*Transformer Decoder

   ```shell
   sh im2latex/im2latex_res31_infer.sh
   ```

   预测结果在[expr_result/im2latex_res31/predict](expr_result/im2latex_res31/predict)中

<!-- Result -->

## 结果




<!-- Pretrain Model -->

## Pretrained Model
coming soon...












