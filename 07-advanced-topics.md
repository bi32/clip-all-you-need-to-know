# 第7章：CLIP进阶主题

## 7.1 CLIP的变体和改进

### 7.1.1 主要CLIP变体对比

```python
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CLIPVariant:
    """CLIP变体配置"""
    name: str
    vision_encoder: str
    text_encoder: str
    embed_dim: int
    improvements: list
    paper_link: str
    
# 主要CLIP变体
clip_variants = {
    "OpenAI-CLIP": CLIPVariant(
        name="Original CLIP",
        vision_encoder="ViT/ResNet",
        text_encoder="Transformer",
        embed_dim=512,
        improvements=["基础版本"],
        paper_link="https://arxiv.org/abs/2103.00020"
    ),
    
    "ALIGN": CLIPVariant(
        name="ALIGN (Google)",
        vision_encoder="EfficientNet",
        text_encoder="BERT",
        embed_dim=640,
        improvements=["更大规模数据（18亿）", "噪声鲁棒性"],
        paper_link="https://arxiv.org/abs/2102.05918"
    ),
    
    "FILIP": CLIPVariant(
        name="FILIP",
        vision_encoder="ViT",
        text_encoder="Transformer",
        embed_dim=768,
        improvements=["细粒度对比学习", "token级别匹配"],
        paper_link="https://arxiv.org/abs/2111.07783"
    ),
    
    "DeCLIP": CLIPVariant(
        name="DeCLIP",
        vision_encoder="ViT",
        text_encoder="Transformer",
        embed_dim=512,
        improvements=["数据高效", "自监督增强"],
        paper_link="https://arxiv.org/abs/2110.05208"
    ),
    
    "SLIP": CLIPVariant(
        name="SLIP",
        vision_encoder="ViT",
        text_encoder="Transformer",
        embed_dim=512,
        improvements=["结合自监督学习", "SimCLR集成"],
        paper_link="https://arxiv.org/abs/2112.12750"
    ),
    
    "CyCLIP": CLIPVariant(
        name="CyCLIP",
        vision_encoder="ViT/ResNet",
        text_encoder="Transformer",
        embed_dim=512,
        improvements=["几何一致性", "循环一致性损失"],
        paper_link="https://arxiv.org/abs/2205.14459"
    ),
    
    "Chinese-CLIP": CLIPVariant(
        name="Chinese-CLIP",
        vision_encoder="ViT",
        text_encoder="RoBERTa",
        embed_dim=512,
        improvements=["中文支持", "多语言能力"],
        paper_link="https://arxiv.org/abs/2211.01335"
    )
}
```

### 7.1.2 FILIP：细粒度对比学习

```python
class FILIP(nn.Module):
    """FILIP: Fine-grained Interactive Language-Image Pre-training"""
    
    def __init__(self, vision_encoder, text_encoder, temperature=0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
    def forward(self, images, texts):
        # 获取patch级别的图像特征
        # [batch_size, num_patches, embed_dim]
        image_features = self.vision_encoder.forward_patches(images)
        
        # 获取token级别的文本特征
        # [batch_size, seq_len, embed_dim]
        text_features = self.text_encoder.forward_tokens(texts)
        
        # 计算细粒度相似度
        similarity = self.compute_fine_grained_similarity(
            image_features, text_features
        )
        
        return similarity
    
    def compute_fine_grained_similarity(self, image_features, text_features):
        """计算细粒度的图像-文本相似度"""
        batch_size = image_features.shape[0]
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算所有patch和token之间的相似度
        # [batch_size, batch_size, num_patches, seq_len]
        similarity_matrix = torch.einsum('bpd,csd->bcps', 
                                        image_features, text_features)
        
        # 对每个图像-文本对，取最大相似度
        # [batch_size, batch_size]
        max_similarity = similarity_matrix.max(dim=-1)[0].max(dim=-1)[0]
        
        return max_similarity / self.temperature
    
    def filip_loss(self, images, texts):
        """FILIP损失函数"""
        similarity = self.forward(images, texts)
        
        batch_size = similarity.shape[0]
        labels = torch.arange(batch_size).to(similarity.device)
        
        # 对比损失
        loss_i2t = F.cross_entropy(similarity, labels)
        loss_t2i = F.cross_entropy(similarity.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2
```

### 7.1.3 CyCLIP：循环一致性

```python
class CyCLIP(nn.Module):
    """CyCLIP: Cyclic Contrastive Language-Image Pretraining"""
    
    def __init__(self, clip_model, cyclic_weight=0.25):
        super().__init__()
        self.clip_model = clip_model
        self.cyclic_weight = cyclic_weight
        
    def cyclic_consistency_loss(self, image_features, text_features):
        """循环一致性损失"""
        batch_size = image_features.shape[0]
        
        # 归一化
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 图像 -> 文本 -> 图像
        i2t_similarity = image_features @ text_features.t()
        i2t_assignment = F.softmax(i2t_similarity, dim=-1)
        reconstructed_image = i2t_assignment @ text_features
        
        # 文本 -> 图像 -> 文本
        t2i_similarity = text_features @ image_features.t()
        t2i_assignment = F.softmax(t2i_similarity, dim=-1)
        reconstructed_text = t2i_assignment @ image_features
        
        # 循环一致性损失
        image_cyclic_loss = F.mse_loss(reconstructed_image, image_features)
        text_cyclic_loss = F.mse_loss(reconstructed_text, text_features)
        
        return (image_cyclic_loss + text_cyclic_loss) / 2
    
    def forward(self, images, texts):
        # 标准CLIP前向传播
        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(texts)
        
        # 计算标准对比损失
        logits_per_image, logits_per_text = self.clip_model(images, texts)
        
        batch_size = images.shape[0]
        labels = torch.arange(batch_size).to(images.device)
        
        contrastive_loss = (F.cross_entropy(logits_per_image, labels) + 
                           F.cross_entropy(logits_per_text, labels)) / 2
        
        # 添加循环一致性损失
        cyclic_loss = self.cyclic_consistency_loss(image_features, text_features)
        
        total_loss = contrastive_loss + self.cyclic_weight * cyclic_loss
        
        return total_loss, {
            'contrastive_loss': contrastive_loss.item(),
            'cyclic_loss': cyclic_loss.item()
        }
```

## 7.2 多模态融合技术

### 7.2.1 跨模态注意力机制

```python
class CrossModalAttention(nn.Module):
    """跨模态注意力机制"""
    
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(self, visual_features, text_features):
        """
        visual_features: [seq_len_v, batch_size, embed_dim]
        text_features: [seq_len_t, batch_size, embed_dim]
        """
        # 视觉特征作为query，文本特征作为key和value
        attn_output, _ = self.multihead_attn(
            visual_features, text_features, text_features
        )
        
        # 残差连接和层归一化
        visual_features = self.layer_norm1(visual_features + attn_output)
        
        # FFN
        ffn_output = self.ffn(visual_features)
        visual_features = self.layer_norm2(visual_features + ffn_output)
        
        return visual_features

class MultiModalFusion(nn.Module):
    """多模态融合模块"""
    
    def __init__(self, embed_dim, num_layers=4):
        super().__init__()
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(embed_dim) for _ in range(num_layers)
        ])
        
        self.fusion_projection = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, image_features, text_features):
        """
        融合图像和文本特征
        """
        # 交替应用跨模态注意力
        for i, layer in enumerate(self.cross_attn_layers):
            if i % 2 == 0:
                # 图像attending到文本
                image_features = layer(image_features, text_features)
            else:
                # 文本attending到图像
                text_features = layer(text_features, image_features)
        
        # 融合特征
        # 取[CLS] token或全局池化
        image_global = image_features.mean(dim=0)  # [batch_size, embed_dim]
        text_global = text_features.mean(dim=0)    # [batch_size, embed_dim]
        
        # 拼接并投影
        fused_features = torch.cat([image_global, text_global], dim=-1)
        fused_features = self.fusion_projection(fused_features)
        
        return fused_features
```

### 7.2.2 模态对齐技术

```python
class ModalityAlignment(nn.Module):
    """模态对齐模块"""
    
    def __init__(self, embed_dim):
        super().__init__()
        # 对齐网络
        self.image_align = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.text_align = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 领域判别器（对抗训练）
        self.domain_discriminator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, image_features, text_features):
        # 对齐特征
        aligned_image = self.image_align(image_features)
        aligned_text = self.text_align(text_features)
        
        # 领域判别（用于对抗训练）
        all_features = torch.cat([aligned_image, aligned_text], dim=0)
        domain_labels = torch.cat([
            torch.ones(aligned_image.shape[0], 1),   # 图像=1
            torch.zeros(aligned_text.shape[0], 1)    # 文本=0
        ], dim=0).to(all_features.device)
        
        domain_predictions = self.domain_discriminator(all_features)
        
        # 对抗损失：让判别器无法区分两种模态
        adversarial_loss = F.binary_cross_entropy(domain_predictions, domain_labels)
        
        return aligned_image, aligned_text, adversarial_loss
    
    def compute_alignment_loss(self, image_features, text_features):
        """计算对齐损失"""
        aligned_image, aligned_text, adv_loss = self.forward(image_features, text_features)
        
        # 最大均值差异（MMD）损失
        mmd_loss = self.mmd_loss(aligned_image, aligned_text)
        
        # 中心损失：让同类特征聚集
        center_loss = self.center_loss(aligned_image, aligned_text)
        
        total_loss = adv_loss + 0.1 * mmd_loss + 0.05 * center_loss
        
        return total_loss
    
    def mmd_loss(self, x, y, kernel='rbf'):
        """最大均值差异损失"""
        xx = self.kernel_func(x, x, kernel)
        yy = self.kernel_func(y, y, kernel)
        xy = self.kernel_func(x, y, kernel)
        
        return xx.mean() + yy.mean() - 2 * xy.mean()
    
    def kernel_func(self, x, y, kernel='rbf'):
        """核函数"""
        if kernel == 'rbf':
            # RBF核
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist ** 2 / (2 * 1.0))
        else:
            # 线性核
            return x @ y.t()
    
    def center_loss(self, image_features, text_features):
        """中心损失"""
        # 计算特征中心
        image_center = image_features.mean(dim=0, keepdim=True)
        text_center = text_features.mean(dim=0, keepdim=True)
        
        # 中心应该接近
        return F.mse_loss(image_center, text_center)
```

## 7.3 CLIP与大语言模型结合

### 7.3.1 CLIP + GPT集成

```python
class CLIPGPTModel(nn.Module):
    """CLIP与GPT结合的模型"""
    
    def __init__(self, clip_model, gpt_model):
        super().__init__()
        self.clip = clip_model
        self.gpt = gpt_model
        
        # 适配层：将CLIP特征映射到GPT的输入空间
        self.visual_adapter = nn.Linear(
            self.clip.visual.output_dim,
            self.gpt.config.hidden_size
        )
        
        # 冻结预训练模型
        for param in self.clip.parameters():
            param.requires_grad = False
            
    def generate_caption(self, image, max_length=50, temperature=1.0):
        """为图像生成描述"""
        # 提取视觉特征
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
        
        # 适配到GPT输入空间
        visual_prefix = self.visual_adapter(image_features)
        visual_prefix = visual_prefix.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # 准备生成
        batch_size = image.shape[0]
        generated_ids = torch.zeros(batch_size, max_length, dtype=torch.long).to(image.device)
        
        # 起始token
        input_ids = torch.tensor([[self.gpt.config.bos_token_id]] * batch_size).to(image.device)
        
        for i in range(max_length):
            # 获取GPT嵌入
            inputs_embeds = self.gpt.transformer.wte(input_ids)
            
            # 拼接视觉前缀
            inputs_embeds = torch.cat([visual_prefix, inputs_embeds], dim=1)
            
            # GPT前向传播
            outputs = self.gpt(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, -1, :] / temperature
            
            # 采样下一个token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated_ids[:, i] = next_token.squeeze()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # 检查是否所有序列都生成了结束token
            if (next_token == self.gpt.config.eos_token_id).all():
                break
        
        return generated_ids
    
    def visual_question_answering(self, image, question, max_answer_length=20):
        """视觉问答"""
        # 编码问题
        question_tokens = self.gpt.tokenizer(question, return_tensors="pt").to(image.device)
        
        # 提取视觉特征
        with torch.no_grad():
            image_features = self.clip.encode_image(image)
        
        visual_prefix = self.visual_adapter(image_features).unsqueeze(1)
        
        # 组合输入：[视觉前缀] + [问题] + [答案开始标记]
        question_embeds = self.gpt.transformer.wte(question_tokens.input_ids)
        
        # 添加答案提示
        answer_prompt = "Answer:"
        answer_tokens = self.gpt.tokenizer(answer_prompt, return_tensors="pt").to(image.device)
        answer_embeds = self.gpt.transformer.wte(answer_tokens.input_ids)
        
        # 拼接所有输入
        full_embeds = torch.cat([visual_prefix, question_embeds, answer_embeds], dim=1)
        
        # 生成答案
        outputs = self.gpt.generate(
            inputs_embeds=full_embeds,
            max_new_tokens=max_answer_length,
            temperature=0.7,
            do_sample=True
        )
        
        return outputs
```

### 7.3.2 CLIP-BERT多模态理解

```python
class CLIPBERTMultiModal(nn.Module):
    """CLIP-BERT多模态理解模型"""
    
    def __init__(self, clip_model, bert_model):
        super().__init__()
        self.clip = clip_model
        self.bert = bert_model
        
        # 模态类型嵌入
        self.modality_embeddings = nn.Embedding(2, bert_model.config.hidden_size)
        
        # 视觉到BERT的投影
        self.visual_projection = nn.Linear(
            clip_model.visual.output_dim,
            bert_model.config.hidden_size
        )
        
    def forward(self, images, input_ids, attention_mask=None):
        """
        多模态前向传播
        """
        # 提取视觉特征
        with torch.no_grad():
            image_features = self.clip.encode_image(images)  # [batch, embed_dim]
        
        # 投影到BERT空间
        visual_embeds = self.visual_projection(image_features)  # [batch, hidden_size]
        visual_embeds = visual_embeds.unsqueeze(1)  # [batch, 1, hidden_size]
        
        # 获取文本嵌入
        text_embeds = self.bert.embeddings.word_embeddings(input_ids)
        
        # 添加模态类型嵌入
        visual_modality = self.modality_embeddings(
            torch.zeros(visual_embeds.shape[:2], dtype=torch.long).to(images.device)
        )
        text_modality = self.modality_embeddings(
            torch.ones(text_embeds.shape[:2], dtype=torch.long).to(images.device)
        )
        
        visual_embeds = visual_embeds + visual_modality
        text_embeds = text_embeds + text_modality
        
        # 拼接视觉和文本嵌入
        combined_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 更新注意力掩码
        if attention_mask is not None:
            visual_mask = torch.ones(visual_embeds.shape[:2]).to(attention_mask.device)
            combined_mask = torch.cat([visual_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # BERT编码
        outputs = self.bert(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask
        )
        
        return outputs
    
    def compute_similarity(self, images, texts):
        """计算图像-文本相似度（用于检索）"""
        # 使用CLIP计算相似度
        with torch.no_grad():
            image_features = self.clip.encode_image(images)
            text_features = self.clip.encode_text(texts)
            
            # 归一化
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            
            # 相似度矩阵
            similarity = image_features @ text_features.t()
        
        return similarity
```

## 7.4 CLIP的可解释性

### 7.4.1 注意力可视化

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CLIPExplainability:
    """CLIP可解释性工具"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        
        # 注册钩子以捕获注意力权重
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子以捕获注意力权重"""
        def hook_fn(module, input, output, name):
            if isinstance(output, tuple):
                self.attention_weights[name] = output[1]  # 注意力权重
            else:
                self.attention_weights[name] = output
        
        # 为视觉编码器的注意力层注册钩子
        for name, module in self.model.visual.named_modules():
            if 'attn' in name.lower():
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
    
    def visualize_attention(self, image, text, layer_idx=-1):
        """可视化注意力图"""
        # 前向传播
        image_tensor = self.preprocess(image).unsqueeze(0)
        text_tokens = clip.tokenize([text])
        
        with torch.no_grad():
            _ = self.model(image_tensor, text_tokens)
        
        # 获取指定层的注意力权重
        attn_weights = list(self.attention_weights.values())[layer_idx]
        
        # 处理注意力权重
        # [batch, num_heads, seq_len, seq_len]
        attn_weights = attn_weights[0].mean(dim=0)  # 平均所有注意力头
        
        # 获取[CLS] token对其他token的注意力
        cls_attn = attn_weights[0, 1:]  # 排除[CLS]自己
        
        # Reshape到图像尺寸
        num_patches = int(np.sqrt(len(cls_attn)))
        attention_map = cls_attn.reshape(num_patches, num_patches)
        
        # 上采样到原始图像尺寸
        attention_map = cv2.resize(
            attention_map.cpu().numpy(),
            (image.size[0], image.size[1]),
            interpolation=cv2.INTER_CUBIC
        )
        
        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # 注意力图
        im = axes[1].imshow(attention_map, cmap='hot')
        axes[1].set_title(f"Attention Map for '{text}'")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
        
        # 叠加显示
        axes[2].imshow(image)
        axes[2].imshow(attention_map, cmap='hot', alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig
    
    def gradcam_visualization(self, image, text, target_layer=None):
        """使用Grad-CAM可视化"""
        image_tensor = self.preprocess(image).unsqueeze(0).requires_grad_(True)
        text_tokens = clip.tokenize([text])
        
        # 前向传播
        image_features = self.model.encode_image(image_tensor)
        text_features = self.model.encode_text(text_tokens)
        
        # 计算相似度
        similarity = (image_features @ text_features.t()).squeeze()
        
        # 反向传播
        self.model.zero_grad()
        similarity.backward()
        
        # 获取目标层的梯度和激活
        if target_layer is None:
            target_layer = self.model.visual.transformer.resblocks[-1]
        
        gradients = image_tensor.grad
        activations = target_layer.output  # 需要在前向传播时保存
        
        # 计算Grad-CAM
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / cam.max()
        
        # 上采样到原始尺寸
        cam = F.interpolate(cam, size=image.size, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        return cam
    
    def token_importance_analysis(self, image, text):
        """分析文本token的重要性"""
        image_tensor = self.preprocess(image).unsqueeze(0)
        tokens = clip.tokenize([text])[0]
        
        token_importance = []
        
        # 逐个遮蔽token并测量影响
        for i in range(len(tokens)):
            if tokens[i] == 0:  # padding token
                break
            
            # 创建遮蔽版本
            masked_tokens = tokens.clone()
            masked_tokens[i] = 0  # 用padding token替换
            
            # 计算相似度
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                
                original_text_features = self.model.encode_text(tokens.unsqueeze(0))
                masked_text_features = self.model.encode_text(masked_tokens.unsqueeze(0))
                
                original_sim = (image_features @ original_text_features.t()).item()
                masked_sim = (image_features @ masked_text_features.t()).item()
                
                importance = original_sim - masked_sim
                token_importance.append(importance)
        
        # 解码token以获取单词
        tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        words = []
        for token_id in tokens[:len(token_importance)]:
            words.append(tokenizer.decode([token_id.item()]))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(words)), token_importance)
        plt.xticks(range(len(words)), words, rotation=45)
        plt.ylabel("Importance Score")
        plt.title("Token Importance Analysis")
        plt.tight_layout()
        
        return dict(zip(words, token_importance))
```

### 7.4.2 特征分析工具

```python
class CLIPFeatureAnalyzer:
    """CLIP特征分析器"""
    
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        
    def analyze_feature_space(self, images, texts):
        """分析特征空间的性质"""
        # 提取特征
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(texts)
            
            # 归一化
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
        
        analysis = {}
        
        # 1. 特征统计
        analysis['image_feature_stats'] = {
            'mean': image_features.mean(dim=0).cpu().numpy(),
            'std': image_features.std(dim=0).cpu().numpy(),
            'sparsity': (image_features == 0).float().mean().item()
        }
        
        analysis['text_feature_stats'] = {
            'mean': text_features.mean(dim=0).cpu().numpy(),
            'std': text_features.std(dim=0).cpu().numpy(),
            'sparsity': (text_features == 0).float().mean().item()
        }
        
        # 2. 模态间对齐度
        alignment_score = F.cosine_similarity(
            image_features.mean(dim=0),
            text_features.mean(dim=0),
            dim=0
        ).item()
        analysis['modality_alignment'] = alignment_score
        
        # 3. 特征多样性（使用平均成对距离）
        image_diversity = self._compute_diversity(image_features)
        text_diversity = self._compute_diversity(text_features)
        
        analysis['image_diversity'] = image_diversity
        analysis['text_diversity'] = text_diversity
        
        # 4. 主成分分析
        analysis['pca'] = self._pca_analysis(image_features, text_features)
        
        return analysis
    
    def _compute_diversity(self, features):
        """计算特征多样性"""
        # 计算所有成对距离
        distances = torch.cdist(features, features, p=2)
        
        # 排除对角线（自己到自己的距离）
        mask = ~torch.eye(len(features), dtype=torch.bool).to(distances.device)
        distances = distances[mask]
        
        return {
            'mean_distance': distances.mean().item(),
            'std_distance': distances.std().item(),
            'min_distance': distances.min().item(),
            'max_distance': distances.max().item()
        }
    
    def _pca_analysis(self, image_features, text_features, n_components=10):
        """PCA分析"""
        from sklearn.decomposition import PCA
        
        # 合并特征
        all_features = torch.cat([image_features, text_features], dim=0)
        all_features = all_features.cpu().numpy()
        
        # PCA
        pca = PCA(n_components=n_components)
        pca.fit(all_features)
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
        }
    
    def probe_linguistic_knowledge(self, texts):
        """探测语言知识"""
        # 测试不同的语言现象
        linguistic_tests = {
            'synonyms': [
                ("a photo of a car", "a photo of an automobile"),
                ("a happy person", "a joyful person")
            ],
            'antonyms': [
                ("a big object", "a small object"),
                ("a hot day", "a cold day")
            ],
            'negation': [
                ("a red car", "not a red car"),
                ("a dog", "not a dog")
            ]
        }
        
        results = {}
        
        for test_type, pairs in linguistic_tests.items():
            similarities = []
            
            for text1, text2 in pairs:
                tokens1 = clip.tokenize([text1])
                tokens2 = clip.tokenize([text2])
                
                with torch.no_grad():
                    features1 = self.model.encode_text(tokens1)
                    features2 = self.model.encode_text(tokens2)
                    
                    features1 = F.normalize(features1, dim=-1)
                    features2 = F.normalize(features2, dim=-1)
                    
                    sim = (features1 @ features2.t()).item()
                    similarities.append(sim)
            
            results[test_type] = {
                'mean_similarity': np.mean(similarities),
                'std_similarity': np.std(similarities),
                'examples': list(zip([p[0] for p in pairs], 
                                    [p[1] for p in pairs], 
                                    similarities))
            }
        
        return results
```

## 7.5 CLIP的未来发展

### 7.5.1 研究方向

```python
# 未来研究方向的示例实现

class FutureCLIPDirections:
    """CLIP未来发展方向的示例"""
    
    @staticmethod
    def video_clip():
        """视频CLIP：处理时序信息"""
        class VideoCLIP(nn.Module):
            def __init__(self, clip_model, temporal_encoder):
                super().__init__()
                self.clip = clip_model
                self.temporal = temporal_encoder
                
            def encode_video(self, video_frames):
                # video_frames: [batch, num_frames, channels, height, width]
                batch_size, num_frames = video_frames.shape[:2]
                
                # 提取每帧的特征
                frame_features = []
                for i in range(num_frames):
                    frame_feat = self.clip.encode_image(video_frames[:, i])
                    frame_features.append(frame_feat)
                
                frame_features = torch.stack(frame_features, dim=1)
                
                # 时序编码
                video_features = self.temporal(frame_features)
                
                return video_features
        
        return VideoCLIP
    
    @staticmethod
    def audio_clip():
        """音频CLIP：多模态扩展到音频"""
        class AudioCLIP(nn.Module):
            def __init__(self, clip_model, audio_encoder):
                super().__init__()
                self.clip = clip_model
                self.audio_encoder = audio_encoder
                self.audio_projection = nn.Linear(
                    audio_encoder.output_dim,
                    clip_model.visual.output_dim
                )
                
            def encode_audio(self, audio_waveform):
                audio_features = self.audio_encoder(audio_waveform)
                audio_features = self.audio_projection(audio_features)
                return F.normalize(audio_features, dim=-1)
        
        return AudioCLIP
    
    @staticmethod
    def efficient_clip():
        """高效CLIP：模型压缩和加速"""
        class EfficientCLIP:
            def __init__(self, original_model):
                self.model = original_model
                
            def quantize(self):
                """量化模型"""
                import torch.quantization as quant
                self.model = quant.quantize_dynamic(
                    self.model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                return self.model
            
            def distill(self, student_model, dataloader):
                """知识蒸馏"""
                criterion = nn.KLDivLoss()
                optimizer = torch.optim.Adam(student_model.parameters())
                
                for images, texts in dataloader:
                    # 教师模型预测
                    with torch.no_grad():
                        teacher_img_feat = self.model.encode_image(images)
                        teacher_txt_feat = self.model.encode_text(texts)
                    
                    # 学生模型预测
                    student_img_feat = student_model.encode_image(images)
                    student_txt_feat = student_model.encode_text(texts)
                    
                    # 蒸馏损失
                    loss = criterion(student_img_feat, teacher_img_feat)
                    loss += criterion(student_txt_feat, teacher_txt_feat)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                return student_model
        
        return EfficientCLIP
    
    @staticmethod
    def compositional_clip():
        """组合CLIP：理解复杂组合概念"""
        class CompositionalCLIP(nn.Module):
            def __init__(self, clip_model):
                super().__init__()
                self.clip = clip_model
                self.composition_module = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=512,
                        nhead=8
                    ),
                    num_layers=2
                )
                
            def encode_compositional_text(self, text_parts):
                """编码组合文本
                text_parts: ["red", "car", "on", "street"]
                """
                # 编码每个部分
                part_features = []
                for part in text_parts:
                    tokens = clip.tokenize([part])
                    features = self.clip.encode_text(tokens)
                    part_features.append(features)
                
                part_features = torch.stack(part_features, dim=1)
                
                # 组合编码
                composed_features = self.composition_module(part_features)
                
                # 聚合
                final_features = composed_features.mean(dim=1)
                
                return F.normalize(final_features, dim=-1)
        
        return CompositionalCLIP
```

### 7.5.2 应用前景

```python
class FutureApplications:
    """CLIP的未来应用场景"""
    
    @staticmethod
    def medical_diagnosis():
        """医疗诊断应用"""
        class MedicalCLIP:
            def __init__(self, clip_model, medical_knowledge_base):
                self.clip = clip_model
                self.knowledge_base = medical_knowledge_base
                
            def diagnose(self, medical_image, symptoms_text):
                # 提取图像特征
                image_features = self.clip.encode_image(medical_image)
                
                # 编码症状描述
                symptom_features = self.clip.encode_text(symptoms_text)
                
                # 查询知识库
                possible_conditions = self.knowledge_base.query(
                    image_features, 
                    symptom_features
                )
                
                return possible_conditions
        
        return MedicalCLIP
    
    @staticmethod
    def autonomous_driving():
        """自动驾驶应用"""
        class DrivingCLIP:
            def __init__(self, clip_model):
                self.clip = clip_model
                self.action_decoder = nn.Linear(512, 4)  # 转向、加速、刹车等
                
            def process_scene(self, camera_image, instruction_text):
                # 理解场景
                scene_features = self.clip.encode_image(camera_image)
                
                # 理解指令
                instruction_features = self.clip.encode_text(instruction_text)
                
                # 融合并决策
                combined = scene_features * instruction_features
                actions = self.action_decoder(combined)
                
                return actions
        
        return DrivingCLIP
    
    @staticmethod
    def educational_assistant():
        """教育助手应用"""
        class EducationalCLIP:
            def __init__(self, clip_model):
                self.clip = clip_model
                
            def explain_concept(self, diagram_image, student_question):
                # 理解图表
                diagram_features = self.clip.encode_image(diagram_image)
                
                # 理解问题
                question_features = self.clip.encode_text(student_question)
                
                # 生成解释
                explanation = self.generate_explanation(
                    diagram_features, 
                    question_features
                )
                
                return explanation
            
            def generate_explanation(self, img_feat, txt_feat):
                # 这里可以接入语言模型生成解释
                pass
        
        return EducationalCLIP
```

## 7.6 总结

本章探讨了CLIP的进阶主题：

1. **CLIP变体**：FILIP、CyCLIP等改进版本
2. **多模态融合**：跨模态注意力、模态对齐技术
3. **大模型结合**：与GPT、BERT的集成
4. **可解释性**：注意力可视化、特征分析
5. **未来发展**：视频、音频扩展，新应用场景

这些进阶内容展示了CLIP技术的深度和广度，以及其在AI领域的巨大潜力。

## 下一步

继续阅读 → [README - 教程总览](./README.md)