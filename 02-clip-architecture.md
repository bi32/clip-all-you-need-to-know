# 第2章：CLIP架构详解

## 2.1 CLIP的整体架构

CLIP采用了双塔架构（Dual-Encoder Architecture），由两个独立的编码器组成：

```
输入图像 → [图像编码器] → 图像特征向量
                              ↓
                          [相似度计算]
                              ↑
输入文本 → [文本编码器] → 文本特征向量
```

### 架构设计理念

1. **模块化设计**：图像和文本编码器相互独立，便于单独优化和替换
2. **共享特征空间**：两个编码器输出到同一个特征空间，使得图像和文本特征可以直接比较
3. **端到端训练**：整个系统可以通过对比学习目标函数进行端到端训练

## 2.2 图像编码器

### 2.2.1 Vision Transformer (ViT) 架构

CLIP主要使用Vision Transformer作为图像编码器，这是其成功的关键因素之一。

```python
# ViT的基本结构
class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16, embed_dim=768):
        self.patch_embed = PatchEmbedding(image_size, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.transformer = TransformerEncoder(num_layers=12)
        self.ln = LayerNorm(embed_dim)
```

### 2.2.2 图像处理流程

1. **图像分块（Patching）**
   - 将输入图像（如224×224）分割成固定大小的块（如16×16）
   - 每个块被视为一个"token"

2. **线性投影**
   - 将每个图像块展平并通过线性层投影到嵌入维度
   - 添加可学习的位置编码

3. **Transformer编码**
   - 添加特殊的[CLS] token用于聚合全局信息
   - 通过多层Transformer块处理
   - 每层包含：多头自注意力 → 层归一化 → FFN → 层归一化

4. **特征提取**
   - 使用[CLS] token的输出作为图像的全局表示
   - 或者对所有patch tokens进行池化

### 2.2.3 ResNet变体

CLIP也提供了基于ResNet的图像编码器变体：

```python
# ResNet编码器的修改
class ModifiedResNet:
    def __init__(self):
        # 主要修改：
        # 1. 使用注意力池化替代平均池化
        # 2. 增加投影头将特征映射到共享空间
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.layers = nn.ModuleList([...])  # ResNet blocks
        self.attnpool = AttentionPool2d(...)
```

## 2.3 文本编码器

### 2.3.1 Transformer架构

CLIP的文本编码器基于Transformer架构，类似于GPT-2但有所修改：

```python
class TextTransformer:
    def __init__(self, vocab_size=49408, context_length=77, embed_dim=512):
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, embed_dim))
        self.transformer = TransformerEncoder(
            width=embed_dim,
            layers=12,
            heads=8
        )
        self.ln_final = LayerNorm(embed_dim)
        self.text_projection = nn.Parameter(torch.empty(embed_dim, projection_dim))
```

### 2.3.2 文本处理流程

1. **分词（Tokenization）**
   - 使用Byte Pair Encoding (BPE)
   - 词汇表大小：49,408
   - 最大序列长度：77 tokens

2. **嵌入层**
   - Token嵌入：将每个token映射到嵌入向量
   - 位置嵌入：添加位置信息

3. **Transformer编码**
   - 12层Transformer块
   - 使用因果掩码（causal mask）
   - 多头自注意力机制

4. **特征提取**
   - 使用[EOS] token的输出作为文本表示
   - 通过投影层映射到共享特征空间

### 2.3.3 关键设计选择

1. **因果掩码 vs 双向注意力**
   - CLIP使用因果掩码（类似GPT）
   - 这允许模型进行自回归生成（虽然主要用于编码）

2. **上下文长度**
   - 77 tokens的限制平衡了性能和效率
   - 对于大多数描述性文本足够

## 2.4 对比学习机制

### 2.4.1 InfoNCE损失函数

CLIP的核心是InfoNCE（Noise Contrastive Estimation）损失：

```python
def clip_loss(image_features, text_features, temperature=0.07):
    # 归一化特征
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # 计算相似度矩阵
    logits = image_features @ text_features.T / temperature
    
    # 对称的对比损失
    labels = torch.arange(len(logits)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2t + loss_t2i) / 2
```

### 2.4.2 温度参数的作用

温度参数τ（temperature）控制相似度分布的锐度：
- **低温度**（如0.01）：使分布更尖锐，模型更确定
- **高温度**（如0.1）：使分布更平滑，模型更保守
- CLIP默认使用τ=0.07

### 2.4.3 批次内负样本

在一个批次中，每个图像-文本对的负样本来自：
- 对于图像：批次中其他所有文本
- 对于文本：批次中其他所有图像

批次大小直接影响负样本数量，CLIP使用32,768的大批次。

## 2.5 模型变体和规模

### 2.5.1 CLIP模型系列

| 模型 | 图像编码器 | 参数量 | 图像分辨率 | 特点 |
|------|-----------|--------|-----------|------|
| CLIP-Base | ViT-B/32 | 86M | 224×224 | 基础版本，速度快 |
| CLIP-Base | ViT-B/16 | 86M | 224×224 | 更细粒度的patch |
| CLIP-Large | ViT-L/14 | 304M | 224×224 | 更大容量 |
| CLIP-Large | ViT-L/14@336px | 304M | 336×336 | 高分辨率输入 |
| CLIP-RN50 | ResNet-50 | 38M | 224×224 | CNN架构 |
| CLIP-RN101 | ResNet-101 | 57M | 224×224 | 更深的CNN |

### 2.5.2 选择合适的模型

选择模型时需要考虑：

1. **精度需求**
   - 高精度：选择ViT-L/14@336px
   - 平衡：选择ViT-B/16
   - 速度优先：选择ViT-B/32或ResNet-50

2. **计算资源**
   - GPU内存限制
   - 推理速度要求
   - 批处理大小

3. **应用场景**
   - 细粒度识别：使用高分辨率模型
   - 实时应用：使用小模型
   - 离线处理：可以使用大模型

## 2.6 特征空间分析

### 2.6.1 共享嵌入空间

CLIP创建了一个共享的嵌入空间，其特点：

1. **维度**：通常512或768维
2. **归一化**：特征向量被L2归一化到单位球面
3. **语义对齐**：相似概念的图像和文本在空间中接近

### 2.6.2 特征可视化

```python
# 使用t-SNE可视化CLIP特征空间
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_clip_space(image_features, text_features, labels):
    # 合并特征
    all_features = np.concatenate([image_features, text_features])
    
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    # 分离图像和文本特征
    n = len(image_features)
    img_2d = features_2d[:n]
    txt_2d = features_2d[n:]
    
    # 绘制
    plt.scatter(img_2d[:, 0], img_2d[:, 1], c='blue', label='Images')
    plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c='red', label='Texts')
    
    # 连接配对的图像和文本
    for i in range(n):
        plt.plot([img_2d[i, 0], txt_2d[i, 0]], 
                [img_2d[i, 1], txt_2d[i, 1]], 'gray', alpha=0.3)
```

## 2.7 注意力机制详解

### 2.7.1 多头自注意力

CLIP中的自注意力机制：

```python
class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # 生成Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        
        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力
        out = (attn @ v).reshape(B, N, C)
        return self.out_proj(out)
```

### 2.7.2 注意力模式分析

CLIP学习到的注意力模式展现了有趣的特性：
- **局部注意力**：相邻patches之间的强连接
- **全局注意力**：[CLS] token与所有patches的连接
- **语义注意力**：关注图像中的重要对象

## 2.8 投影头设计

### 2.8.1 线性投影

将编码器输出映射到共享空间：

```python
class ProjectionHead:
    def __init__(self, input_dim, output_dim):
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        
    def forward(self, x):
        # 简单的线性投影
        x = self.projection(x)
        # L2归一化
        return F.normalize(x, dim=-1)
```

### 2.8.2 为什么需要投影头？

1. **维度对齐**：图像和文本编码器可能有不同的输出维度
2. **特征解耦**：分离模态特定特征和共享语义特征
3. **优化灵活性**：可以独立调整投影维度

## 2.9 架构优化技巧

### 2.9.1 混合精度训练

使用FP16进行前向传播，FP32进行梯度累积：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    image_features = image_encoder(images)
    text_features = text_encoder(texts)
    loss = clip_loss(image_features, text_features)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2.9.2 梯度检查点

减少内存使用：

```python
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer:
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x)  # 节省激活内存
        return x
```

### 2.9.3 高效注意力

使用Flash Attention等优化：
- 减少内存访问
- 提高计算效率
- 支持更长序列

## 2.10 总结

CLIP的架构设计体现了多个关键创新：

1. **双塔架构**：独立编码不同模态，灵活且高效
2. **Transformer为主**：统一的架构处理图像和文本
3. **对比学习**：简单有效的训练目标
4. **大规模训练**：利用海量数据学习通用表示

这些设计选择共同造就了CLIP的成功。在下一章中，我们将详细介绍如何训练和测试CLIP模型。

## 下一步

继续阅读 → [第3章：训练和测试流程](./03-training-and-testing.md)