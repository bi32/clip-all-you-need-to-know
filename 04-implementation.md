# 第4章：CLIP代码实现详解

## 4.1 从零开始实现CLIP

### 4.1.1 完整的CLIP模型实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

class CLIP(nn.Module):
    """完整的CLIP模型实现"""
    
    def __init__(self,
                 embed_dim: int = 512,
                 # Vision
                 image_resolution: int = 224,
                 vision_layers: int = 12,
                 vision_width: int = 768,
                 vision_patch_size: int = 32,
                 # Text
                 context_length: int = 77,
                 vocab_size: int = 49408,
                 transformer_width: int = 512,
                 transformer_heads: int = 8,
                 transformer_layers: int = 12):
        super().__init__()
        
        self.context_length = context_length
        
        # Vision Model
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim
        )
        
        # Text Model
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        
        # Projection layers
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.initialize_parameters()
    
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    
    def build_attention_mask(self):
        # 创建因果注意力掩码
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 上三角矩阵
        return mask
    
    def encode_image(self, image):
        return self.visual(image)
    
    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # 取出[EOS] token的特征 (最高索引处)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x
    
    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 余弦相似度作为logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        return logits_per_image, logits_per_text
```

### 4.1.2 Vision Transformer实现

```python
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, 
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, 
                               kernel_size=patch_size, stride=patch_size, bias=False)
        
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads)
        
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # 添加CLS token
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
            ), x
        ], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.ln_post(x[:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        
        return x
```

### 4.1.3 Transformer块实现

```python
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) 
                                         for _ in range(layers)])
    
    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
    
    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
    
    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
```

## 4.2 使用OpenAI的CLIP

### 4.2.1 安装和基本使用

```python
# 安装
# pip install git+https://github.com/openai/CLIP.git

import clip
import torch
from PIL import Image

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 准备输入
image = preprocess(Image.open("cat.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)

# 前向传播
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # 计算相似度
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # 输出: [[0.002, 0.003, 0.995]]
```

### 4.2.2 批量处理

```python
class CLIPBatchProcessor:
    def __init__(self, model_name="ViT-B/32", batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.batch_size = batch_size
        
    def process_images(self, image_paths):
        """批量处理图像"""
        all_features = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_images = []
            
            for path in batch_paths:
                image = Image.open(path)
                image = self.preprocess(image)
                batch_images.append(image)
            
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
    
    def process_texts(self, texts):
        """批量处理文本"""
        all_features = []
        
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_tokens = clip.tokenize(batch_texts, truncate=True).to(self.device)
            
            with torch.no_grad():
                features = self.model.encode_text(batch_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(features.cpu())
        
        return torch.cat(all_features, dim=0)
```

## 4.3 使用Hugging Face Transformers

### 4.3.1 基本使用

```python
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

# 加载模型和处理器
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 准备输入
image = Image.open("cat.jpg")
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# 处理输入
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 获取输出
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print(f"Probabilities: {probs}")
```

### 4.3.2 高级功能

```python
from transformers import CLIPModel, CLIPTokenizer, CLIPImageProcessor
import torch

class HuggingFaceCLIP:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # 移到GPU
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
    
    def get_image_features(self, images):
        """提取图像特征"""
        inputs = self.image_processor(images=images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        
        return image_features
    
    def get_text_features(self, texts):
        """提取文本特征"""
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(self, images, texts):
        """计算图像-文本相似度"""
        image_features = self.get_image_features(images)
        text_features = self.get_text_features(texts)
        
        # 计算余弦相似度
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        return similarity
```

## 4.4 实用工具类

### 4.4.1 图像搜索引擎

```python
import faiss
import numpy as np
from typing import List, Tuple

class CLIPImageSearchEngine:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.index = None
        self.image_paths = []
        
    def build_index(self, image_paths: List[str], batch_size: int = 32):
        """构建图像索引"""
        print(f"Building index for {len(image_paths)} images...")
        
        # 提取所有图像特征
        all_features = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if batch_images:
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                with torch.no_grad():
                    features = self.model.encode_image(batch_tensor)
                    features = features / features.norm(dim=-1, keepdim=True)
                    all_features.append(features.cpu().numpy())
            
            print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)}")
        
        # 合并所有特征
        features_matrix = np.vstack(all_features).astype('float32')
        
        # 创建FAISS索引
        self.index = faiss.IndexFlatIP(features_matrix.shape[1])  # 内积索引
        self.index.add(features_matrix)
        self.image_paths = image_paths
        
        print(f"Index built with {self.index.ntotal} images")
    
    def search_by_text(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """使用文本查询搜索图像"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # 编码查询文本
        text = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype('float32')
        
        # 搜索
        scores, indices = self.index.search(text_features, k)
        
        # 返回结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(score)))
        
        return results
    
    def search_by_image(self, image_path: str, k: int = 10) -> List[Tuple[str, float]]:
        """使用图像查询搜索相似图像"""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # 编码查询图像
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().numpy().astype('float32')
        
        # 搜索
        scores, indices = self.index.search(image_features, k)
        
        # 返回结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(score)))
        
        return results
    
    def save_index(self, path: str):
        """保存索引到文件"""
        faiss.write_index(self.index, f"{path}/clip.index")
        
        # 保存图像路径
        with open(f"{path}/image_paths.txt", 'w') as f:
            for path in self.image_paths:
                f.write(f"{path}\n")
    
    def load_index(self, path: str):
        """从文件加载索引"""
        self.index = faiss.read_index(f"{path}/clip.index")
        
        # 加载图像路径
        with open(f"{path}/image_paths.txt", 'r') as f:
            self.image_paths = [line.strip() for line in f]
```

### 4.4.2 零样本分类器

```python
class ZeroShotClassifier:
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.class_embeddings = None
        self.class_names = None
    
    def setup_classes(self, class_names: List[str], templates: List[str] = None):
        """设置分类类别"""
        if templates is None:
            templates = [
                'a photo of a {}.',
                'a blurry photo of a {}.',
                'a black and white photo of a {}.',
                'a low contrast photo of a {}.',
                'a high contrast photo of a {}.',
                'a bad photo of a {}.',
                'a good photo of a {}.',
                'a photo of a small {}.',
                'a photo of a big {}.',
            ]
        
        self.class_names = class_names
        
        # 为每个类别创建文本嵌入
        class_embeddings = []
        
        for classname in class_names:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(texts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # 平均多个模板的嵌入
                class_embedding = text_features.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                class_embeddings.append(class_embedding)
        
        self.class_embeddings = torch.stack(class_embeddings, dim=1).to(self.device)
    
    def classify(self, image_path: str, top_k: int = 5):
        """对单张图像进行分类"""
        if self.class_embeddings is None:
            raise ValueError("Classes not set up. Call setup_classes first.")
        
        # 处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 提取图像特征
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (100.0 * image_features @ self.class_embeddings)
            probs = similarity.softmax(dim=-1)
        
        # 获取top-k预测
        values, indices = probs[0].topk(top_k)
        
        predictions = []
        for value, index in zip(values, indices):
            predictions.append({
                'class': self.class_names[index],
                'confidence': float(value)
            })
        
        return predictions
    
    def classify_batch(self, image_paths: List[str], top_k: int = 5):
        """批量分类"""
        if self.class_embeddings is None:
            raise ValueError("Classes not set up. Call setup_classes first.")
        
        # 批量处理图像
        images = []
        valid_paths = []
        
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                image = self.preprocess(image)
                images.append(image)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue
        
        if not images:
            return []
        
        batch_tensor = torch.stack(images).to(self.device)
        
        # 提取特征并分类
        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            similarity = (100.0 * image_features @ self.class_embeddings)
            probs = similarity.softmax(dim=-1)
        
        # 收集结果
        results = []
        for i, path in enumerate(valid_paths):
            values, indices = probs[i].topk(top_k)
            predictions = []
            
            for value, index in zip(values, indices):
                predictions.append({
                    'class': self.class_names[index],
                    'confidence': float(value)
                })
            
            results.append({
                'image': path,
                'predictions': predictions
            })
        
        return results
```

## 4.5 性能优化实现

### 4.5.1 TorchScript优化

```python
def optimize_with_torchscript(model):
    """使用TorchScript优化模型"""
    model.eval()
    
    # 示例输入
    example_image = torch.randn(1, 3, 224, 224).cuda()
    example_text = torch.randint(0, 49408, (1, 77)).cuda()
    
    # 跟踪模型
    traced_model = torch.jit.trace(model, (example_image, example_text))
    
    # 优化
    traced_model = torch.jit.optimize_for_inference(traced_model)
    
    return traced_model

# 使用优化后的模型
optimized_model = optimize_with_torchscript(model)

# 推理
with torch.no_grad():
    image_features = optimized_model.encode_image(images)
    text_features = optimized_model.encode_text(texts)
```

### 4.5.2 ONNX导出和优化

```python
import onnx
import onnxruntime as ort

def export_to_onnx(model, save_path):
    """导出模型到ONNX格式"""
    model.eval()
    
    # 准备示例输入
    dummy_image = torch.randn(1, 3, 224, 224).cuda()
    dummy_text = torch.randint(0, 49408, (1, 77)).cuda()
    
    # 导出
    torch.onnx.export(
        model,
        (dummy_image, dummy_text),
        save_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['image', 'text'],
        output_names=['image_features', 'text_features'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'text': {0: 'batch_size'}
        }
    )
    
    # 验证模型
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {save_path}")

class ONNXCLIPInference:
    def __init__(self, onnx_path):
        # 创建推理会话
        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        
    def encode_image(self, images):
        """使用ONNX推理图像编码"""
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: images.numpy()})
        return torch.from_numpy(output[0])
    
    def encode_text(self, texts):
        """使用ONNX推理文本编码"""
        input_name = self.session.get_inputs()[1].name
        output = self.session.run(None, {input_name: texts.numpy()})
        return torch.from_numpy(output[1])
```

### 4.5.3 量化优化

```python
import torch.quantization as quantization

def quantize_model(model):
    """动态量化模型以减少内存使用和加速推理"""
    # 设置量化配置
    model.eval()
    
    # 动态量化
    quantized_model = quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    
    return quantized_model

# 静态量化（需要校准数据）
def static_quantize_model(model, calibration_loader):
    """静态量化需要校准数据"""
    model.eval()
    
    # 准备量化
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    quantization.prepare(model, inplace=True)
    
    # 校准
    with torch.no_grad():
        for images, texts in calibration_loader:
            _ = model(images, texts)
    
    # 转换为量化模型
    quantization.convert(model, inplace=True)
    
    return model
```

## 4.6 实际应用示例

### 4.6.1 图像标注生成器

```python
class ImageCaptioner:
    def __init__(self, clip_model="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device)
        
        # 预定义的标注模板
        self.templates = {
            'object': ['a {} in the image', 'there is a {}', 'a photo of a {}'],
            'scene': ['the scene shows {}', 'this is a {}', 'a {} environment'],
            'action': ['someone is {}', 'the person is {}', '{} is happening'],
            'attribute': ['the image is {}', 'this looks {}', 'a {} photo']
        }
        
        # 候选词汇
        self.vocabularies = {
            'object': ['person', 'car', 'dog', 'cat', 'building', 'tree', 'flower'],
            'scene': ['indoor', 'outdoor', 'street', 'nature', 'city', 'beach', 'mountain'],
            'action': ['walking', 'running', 'sitting', 'standing', 'playing', 'working'],
            'attribute': ['bright', 'dark', 'colorful', 'monochrome', 'clear', 'blurry']
        }
    
    def generate_caption(self, image_path):
        """为图像生成描述"""
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        caption_parts = []
        
        for category, words in self.vocabularies.items():
            templates = self.templates[category]
            
            # 为每个词汇创建候选句子
            candidates = []
            for word in words:
                for template in templates:
                    candidates.append(template.format(word))
            
            # 编码候选句子
            text = clip.tokenize(candidates).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image)
                text_features = self.clip_model.encode_text(text)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # 选择最佳匹配
                best_idx = similarity[0].argmax().item()
                best_caption = candidates[best_idx]
                best_score = similarity[0][best_idx].item()
                
                if best_score > 0.2:  # 阈值过滤
                    caption_parts.append(best_caption)
        
        # 组合描述
        if caption_parts:
            return ". ".join(caption_parts) + "."
        else:
            return "An image."
```

### 4.6.2 视觉问答系统

```python
class VisualQA:
    def __init__(self, clip_model="ViT-L/14"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(clip_model, device=self.device)
        
    def answer_question(self, image_path, question, candidate_answers):
        """回答关于图像的问题"""
        # 处理图像
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 创建问题-答案对
        qa_pairs = [f"{question} {answer}" for answer in candidate_answers]
        text = clip.tokenize(qa_pairs).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            
            # 归一化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (100.0 * image_features @ text_features.T)
            probs = similarity.softmax(dim=-1)
        
        # 获取最佳答案
        best_idx = probs[0].argmax().item()
        confidence = probs[0][best_idx].item()
        
        return {
            'answer': candidate_answers[best_idx],
            'confidence': float(confidence),
            'all_scores': {
                answer: float(prob) 
                for answer, prob in zip(candidate_answers, probs[0])
            }
        }
    
    def multi_choice_qa(self, image_path, questions_and_options):
        """处理多个选择题"""
        results = []
        
        for item in questions_and_options:
            question = item['question']
            options = item['options']
            
            answer = self.answer_question(image_path, question, options)
            results.append({
                'question': question,
                'answer': answer['answer'],
                'confidence': answer['confidence']
            })
        
        return results
```

## 4.7 总结

本章详细介绍了CLIP的代码实现，包括：

1. **从零实现**：完整的CLIP模型架构实现
2. **使用现有库**：OpenAI CLIP和Hugging Face的使用方法
3. **实用工具**：图像搜索引擎、零样本分类器等
4. **性能优化**：TorchScript、ONNX、量化等优化技术
5. **应用示例**：图像标注、视觉问答等实际应用

掌握这些实现细节后，你可以根据具体需求定制和优化CLIP模型。

## 下一步

继续阅读 → [第5章：CLIP应用场景和玩法](./05-applications.md)