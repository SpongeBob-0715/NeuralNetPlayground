# 三层神经网络实现的 CIFAR-10 图像分类器
本项目基于 NumPy 手工实现了一个三层神经网络，用于在 CIFAR-10 数据集上完成图像分类任务。不依赖 PyTorch / TensorFlow 等自动微分框架。
模型权重下载地址：https://pan.baidu.com/s/1UhLgHyFwjUsqxrpv5gvHiw?pwd=i46g

## 功能

- **数据加载**：高效加载和预处理 CIFAR-10 数据集。
- **模型**：实现一个可定制的三层神经网络，支持 ReLU/Sigmoid 激活函数和 Dropout。
- **训练**：支持小批量 SGD 训练，包含学习率衰减和早停机制。
- **参数搜索**：对超参数进行网格搜索，寻找最佳模型。
- **可视化**：生成权重、偏置、激活值和 t-SNE 嵌入的详细可视化图表。
- **模块化设计**：代码分为数据处理、模型、训练和可视化等可重用模块。

## 🚀 使用方法

### 1️⃣ 安装依赖

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TQDM

通过 pip 安装依赖：
```bash
pip install numpy matplotlib seaborn scikit-learn tqdm
```



### 2️⃣ 数据准备
从 CIFAR-10官方下载数据集

data_batch_1 ~ data_batch_5
test_batch
batches.meta
或者你也可以使用我已经准备好的 data_loader.py 直接加载本地数据。

### 3️⃣ 使用方法
- Training: `python src/main.py --train`
- Testing: `python src/main.py --test`
- Parameter Search: `python src/main.py --param_search`
- Visualization: `bash run_script.sh`

### 4️⃣ Parameter Search可选参数说明
hidden_size：隐藏层神经元个数，默认 128 可选 [128, 256, 512, 1024]

learning_rate：学习率，默认 0.01 可选 [0.1, 0.01, 0.001]

reg：L2正则化系数，默认 0.01

dropout_rate：Dropout比例，默认 0

activation：激活函数类型，可选 relu 或 sigmoid

### 5️⃣ 可视化模块
layer1_weights.png

layer2_weights_heatmap.png

bias_distributions.png

sample_activations.png

tsne_hidden.png
