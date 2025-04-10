# ThreeLayerNetLab

**ThreeLayerNetLab** 是一个基于 Python 的深度学习项目，作为深度学习课程的第一次作业。它实现了一个三层神经网络，用于对 CIFAR-10 数据集中的图像进行分类，并提供训练、测试、参数搜索和模型参数可视化的功能。

## 功能

- **数据加载**：高效加载和预处理 CIFAR-10 数据集。
- **模型**：实现一个可定制的三层神经网络，支持 ReLU/Sigmoid 激活函数和 Dropout。
- **训练**：支持小批量 SGD 训练，包含学习率衰减和早停机制。
- **参数搜索**：对超参数进行网格搜索，寻找最佳模型。
- **可视化**：生成权重、偏置、激活值和 t-SNE 嵌入的详细可视化图表。
- **模块化设计**：代码分为数据处理、模型、训练和可视化等可重用模块。

## 依赖

- Python 3.8+
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- TQDM

通过 pip 安装依赖：
```bash
pip install numpy matplotlib seaborn scikit-learn tqdm
