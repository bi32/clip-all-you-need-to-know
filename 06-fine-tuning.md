# 第6章：CLIP微调教程和资源需求

## 6.1 为什么要微调CLIP？

### 6.1.1 微调的应用场景

虽然CLIP具有强大的零样本能力，但在以下场景中微调可以显著提升性能：

1. **领域特定任务**
   - 医学图像分析（X光、CT、MRI）
   - 卫星图像解析
   - 工业缺陷检测
   - 艺术风格识别

2. **细粒度分类**
   - 区分相似的子类别（如不同品种的狗）
   - 产品型号识别
   - 人脸识别和属性分析

3. **自定义概念学习**
   - 公司特定的产品分类
   - 专有术语和概念
   - 新颖的视觉概念

4. **性能优化**
   - 提高特定任务的准确率
   - 减小模型尺寸（知识蒸馏）
   - 加快推理速度

### 6.1.2 微调 vs 从头训练

| 方面 | 微调CLIP | 从头训练 |
|------|---------|---------|
| 数据需求 | 少量（几千到几万） | 大量（百万级） |
| 训练时间 | 几小时到几天 | 几周到几个月 |
| 计算资源 | 1-8 GPU | 数百个GPU |
| 效果 | 快速收敛，效果好 | 需要大量调优 |
| 成本 | 低 | 高 |

## 6.2 微调策略

### 6.2.1 不同的微调方法

```python
import torch
import torch.nn as nn
from typing import Optional

class CLIPFineTuningStrategies:
    """CLIP微调策略集合"""
    
    @staticmethod
    def linear_probe(clip_model, num_classes: int):
        """
        线性探测：冻结CLIP，只训练一个线性分类器
        适用于：数据量小，任务简单
        """
        # 冻结CLIP参数
        for param in clip_model.parameters():
            param.requires_grad = False
        
        # 添加线性分类头
        class LinearProbeModel(nn.Module):
            def __init__(self, clip_model, num_classes):
                super().__init__()
                self.clip_model = clip_model
                self.classifier = nn.Linear(clip_model.visual.output_dim, num_classes)
                
            def forward(self, images):
                with torch.no_grad():
                    features = self.clip_model.encode_image(images)
                return self.classifier(features)
        
        return LinearProbeModel(clip_model, num_classes)
    
    @staticmethod
    def full_fine_tuning(clip_model, learning_rate: float = 1e-5):
        """
        完全微调：解冻所有参数
        适用于：数据量大，任务复杂
        """
        # 解冻所有参数
        for param in clip_model.parameters():
            param.requires_grad = True
        
        # 使用较小的学习率
        optimizer = torch.optim.AdamW(clip_model.parameters(), lr=learning_rate)
        return clip_model, optimizer
    
    @staticmethod
    def partial_fine_tuning(clip_model, unfreeze_layers: int = 2):
        """
        部分微调：只解冻最后几层
        适用于：中等数据量，平衡效果和效率
        """
        # 冻结大部分层
        for param in clip_model.parameters():
            param.requires_grad = False
        
        # 解冻视觉编码器的最后几层
        if hasattr(clip_model.visual, 'transformer'):
            # ViT架构
            layers = clip_model.visual.transformer.resblocks
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # ResNet架构
            layers = [clip_model.visual.layer4, clip_model.visual.attnpool]
            for layer in layers[-unfreeze_layers:]:
                if layer is not None:
                    for param in layer.parameters():
                        param.requires_grad = True
        
        # 解冻投影层
        if hasattr(clip_model.visual, 'proj'):
            clip_model.visual.proj.requires_grad = True
        
        return clip_model
    
    @staticmethod
    def lora_fine_tuning(clip_model, rank: int = 4):
        """
        LoRA微调：低秩适应
        适用于：参数高效微调
        """
        class LoRALayer(nn.Module):
            def __init__(self, in_features, out_features, rank=4):
                super().__init__()
                self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
                self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
                self.scaling = 0.01
                
            def forward(self, x, weight):
                # 原始权重 + LoRA调整
                lora_weight = (self.lora_B @ self.lora_A) * self.scaling
                return F.linear(x, weight + lora_weight)
        
        # 为注意力层添加LoRA
        for name, module in clip_model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # 替换注意力投影
                lora_layer = LoRALayer(
                    module.embed_dim, 
                    module.embed_dim, 
                    rank=rank
                )
                # 注入LoRA层
                module.register_buffer('lora_layer', lora_layer)
        
        return clip_model
    
    @staticmethod
    def prompt_tuning(clip_model, num_prompts: int = 4):
        """
        提示学习：学习软提示
        适用于：极少量参数调整
        """
        class PromptLearner(nn.Module):
            def __init__(self, clip_model, num_prompts=4):
                super().__init__()
                self.clip_model = clip_model
                
                # 冻结CLIP
                for param in clip_model.parameters():
                    param.requires_grad = False
                
                # 可学习的提示嵌入
                embed_dim = clip_model.transformer.width
                self.prompt_embeddings = nn.Parameter(
                    torch.randn(num_prompts, embed_dim) * 0.02
                )
                
            def forward(self, images, text_tokens):
                # 在文本前添加可学习的提示
                batch_size = text_tokens.shape[0]
                prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 组合提示和原始文本
                text_embeddings = self.clip_model.token_embedding(text_tokens)
                combined = torch.cat([prompts, text_embeddings], dim=1)
                
                # 继续CLIP的前向传播
                return self.clip_model(images, combined)
        
        return PromptLearner(clip_model, num_prompts)
```

### 6.2.2 选择合适的微调策略

```python
def select_fine_tuning_strategy(
    dataset_size: int,
    task_similarity: float,  # 与预训练任务的相似度 (0-1)
    computational_budget: str,  # 'low', 'medium', 'high'
    target_performance: str  # 'baseline', 'good', 'best'
) -> str:
    """
    根据条件推荐微调策略
    """
    if dataset_size < 1000:
        if computational_budget == 'low':
            return "linear_probe"
        else:
            return "prompt_tuning"
    
    elif dataset_size < 10000:
        if task_similarity > 0.7:
            return "partial_fine_tuning"
        else:
            if computational_budget == 'high':
                return "full_fine_tuning"
            else:
                return "lora_fine_tuning"
    
    else:  # dataset_size >= 10000
        if target_performance == 'best':
            return "full_fine_tuning"
        elif target_performance == 'good':
            return "partial_fine_tuning"
        else:
            return "linear_probe"
```

## 6.3 微调实战

### 6.3.1 准备数据集

```python
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path

class CustomCLIPDataset(Dataset):
    """自定义CLIP微调数据集"""
    
    def __init__(self, data_dir: str, split: str = 'train', preprocess=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.preprocess = preprocess
        
        # 加载数据标注
        with open(self.data_dir / f'{split}.json', 'r') as f:
            self.annotations = json.load(f)
        
        # 构建类别映射
        self.classes = sorted(list(set([ann['label'] for ann in self.annotations])))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图像
        image_path = self.data_dir / 'images' / ann['image']
        image = Image.open(image_path).convert('RGB')
        
        if self.preprocess:
            image = self.preprocess(image)
        
        # 文本描述
        text = ann.get('caption', f"a photo of {ann['label']}")
        
        # 类别标签（用于监督学习）
        label = self.class_to_idx[ann['label']]
        
        return {
            'image': image,
            'text': text,
            'label': label,
            'image_path': str(image_path)
        }

def create_data_loaders(data_dir: str, preprocess, batch_size: int = 32):
    """创建数据加载器"""
    train_dataset = CustomCLIPDataset(data_dir, 'train', preprocess)
    val_dataset = CustomCLIPDataset(data_dir, 'val', preprocess)
    test_dataset = CustomCLIPDataset(data_dir, 'test', preprocess)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
```

### 6.3.2 微调训练循环

```python
import clip
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import wandb

class CLIPFineTuner:
    """CLIP微调器"""
    
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.scaler = GradScaler()
        
    def fine_tune(
        self,
        train_loader,
        val_loader,
        strategy: str = "partial",
        num_epochs: int = 10,
        learning_rate: float = 1e-5,
        warmup_steps: int = 500,
        use_wandb: bool = True
    ):
        """执行微调"""
        
        # 应用微调策略
        if strategy == "full":
            self._apply_full_fine_tuning()
        elif strategy == "partial":
            self._apply_partial_fine_tuning()
        elif strategy == "linear":
            self._apply_linear_probe()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 设置优化器
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 学习率调度器
        scheduler = self._get_scheduler(optimizer, warmup_steps, num_epochs * len(train_loader))
        
        # 初始化wandb
        if use_wandb:
            wandb.init(project="clip-finetuning", config={
                "strategy": strategy,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs
            })
        
        # 训练循环
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss, train_acc = self._train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            
            # 验证阶段
            val_loss, val_acc = self._validate(val_loader)
            
            # 日志记录
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")
            
            if use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "learning_rate": scheduler.get_last_lr()[0]
                })
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
        
        return best_val_acc
    
    def _train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            texts = clip.tokenize(batch['text'], truncate=True).to(self.device)
            labels = batch['label'].to(self.device)
            
            # 混合精度训练
            with autocast():
                # 前向传播
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算损失
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
                # 对比学习损失
                batch_size = images.shape[0]
                ground_truth = torch.arange(batch_size).to(self.device)
                loss_i2t = F.cross_entropy(logits, ground_truth)
                loss_t2i = F.cross_entropy(logits.t(), ground_truth)
                
                # 如果有类别标签，添加分类损失
                if self.classification_head is not None:
                    class_logits = self.classification_head(image_features)
                    loss_cls = F.cross_entropy(class_logits, labels)
                    loss = (loss_i2t + loss_t2i) / 2 + loss_cls
                else:
                    loss = (loss_i2t + loss_t2i) / 2
            
            # 反向传播
            optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # 更新权重
            self.scaler.step(optimizer)
            self.scaler.update()
            scheduler.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = logits.max(1)
            correct += predicted.eq(ground_truth).sum().item()
            total += batch_size
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(train_loader), correct / total
    
    def _validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images = batch['image'].to(self.device)
                texts = clip.tokenize(batch['text'], truncate=True).to(self.device)
                
                # 前向传播
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(texts)
                
                # 归一化
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算损失
                logit_scale = self.model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()
                
                batch_size = images.shape[0]
                ground_truth = torch.arange(batch_size).to(self.device)
                loss_i2t = F.cross_entropy(logits, ground_truth)
                loss_t2i = F.cross_entropy(logits.t(), ground_truth)
                loss = (loss_i2t + loss_t2i) / 2
                
                total_loss += loss.item()
                
                # 计算准确率
                _, predicted = logits.max(1)
                correct += predicted.eq(ground_truth).sum().item()
                total += batch_size
        
        return total_loss / len(val_loader), correct / total
    
    def _apply_partial_fine_tuning(self, unfreeze_layers: int = 2):
        """应用部分微调"""
        # 冻结大部分参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻最后几层
        if hasattr(self.model.visual, 'transformer'):
            layers = self.model.visual.transformer.resblocks
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 解冻文本编码器的最后几层
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.resblocks
            for layer in layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # 解冻投影层
        if hasattr(self.model, 'text_projection'):
            self.model.text_projection.requires_grad = True
        if hasattr(self.model.visual, 'proj'):
            self.model.visual.proj.requires_grad = True
        
        # 解冻logit_scale
        self.model.logit_scale.requires_grad = True
        
        self.classification_head = None
    
    def _apply_full_fine_tuning(self):
        """应用完全微调"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.classification_head = None
    
    def _apply_linear_probe(self, num_classes: int = None):
        """应用线性探测"""
        # 冻结所有CLIP参数
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 添加分类头（如果需要）
        if num_classes:
            self.classification_head = nn.Linear(
                self.model.visual.output_dim, 
                num_classes
            ).to(self.device)
        else:
            self.classification_head = None
    
    def _get_scheduler(self, optimizer, warmup_steps, total_steps):
        """获取学习率调度器"""
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def _save_checkpoint(self, epoch, val_acc):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc
        }
        torch.save(checkpoint, f'clip_finetuned_epoch{epoch}_acc{val_acc:.4f}.pt')
```

## 6.4 资源需求分析

### 6.4.1 硬件需求

```python
def estimate_hardware_requirements(
    model_size: str,
    batch_size: int,
    sequence_length: int = 77,
    image_resolution: int = 224,
    fine_tuning_strategy: str = "full"
) -> Dict:
    """
    估算硬件需求
    """
    # 模型参数量（百万）
    model_params = {
        "ViT-B/32": 86,
        "ViT-B/16": 86,
        "ViT-L/14": 304,
        "ViT-L/14@336px": 304,
        "RN50": 38,
        "RN101": 57
    }
    
    # 基础内存需求（GB）
    base_memory = {
        "ViT-B/32": 2,
        "ViT-B/16": 2,
        "ViT-L/14": 5,
        "ViT-L/14@336px": 7,
        "RN50": 1.5,
        "RN101": 2
    }
    
    params = model_params.get(model_size, 86)
    base_mem = base_memory.get(model_size, 2)
    
    # 计算内存需求
    # 模型权重
    model_memory = params * 4 / 1024  # FP32, MB to GB
    
    # 梯度（如果微调）
    if fine_tuning_strategy == "full":
        gradient_memory = model_memory
    elif fine_tuning_strategy == "partial":
        gradient_memory = model_memory * 0.3
    else:  # linear probe
        gradient_memory = 0.01
    
    # 优化器状态（Adam需要2倍模型参数）
    optimizer_memory = gradient_memory * 2
    
    # 激活值内存（与batch size成正比）
    activation_memory = batch_size * image_resolution * image_resolution * 3 * 4 / (1024**3)
    activation_memory += batch_size * sequence_length * 512 * 4 / (1024**3)
    
    # 总内存需求
    total_memory = base_mem + model_memory + gradient_memory + optimizer_memory + activation_memory
    
    # 推荐配置
    recommendations = {
        "minimum_gpu_memory": f"{int(np.ceil(total_memory))} GB",
        "recommended_gpu_memory": f"{int(np.ceil(total_memory * 1.5))} GB",
        "estimated_training_time": estimate_training_time(params, batch_size),
        "recommended_gpus": recommend_gpus(total_memory)
    }
    
    return {
        "model_parameters": f"{params}M",
        "memory_breakdown": {
            "model_weights": f"{model_memory:.2f} GB",
            "gradients": f"{gradient_memory:.2f} GB",
            "optimizer_states": f"{optimizer_memory:.2f} GB",
            "activations": f"{activation_memory:.2f} GB",
            "total": f"{total_memory:.2f} GB"
        },
        "recommendations": recommendations
    }

def recommend_gpus(memory_required: float) -> List[str]:
    """推荐合适的GPU"""
    gpu_memory = {
        "RTX 3060": 12,
        "RTX 3070": 8,
        "RTX 3080": 10,
        "RTX 3090": 24,
        "RTX 4070": 12,
        "RTX 4080": 16,
        "RTX 4090": 24,
        "A10": 24,
        "A40": 48,
        "A100-40GB": 40,
        "A100-80GB": 80,
        "V100": 32,
        "T4": 16
    }
    
    suitable_gpus = []
    for gpu, mem in gpu_memory.items():
        if mem >= memory_required:
            suitable_gpus.append(f"{gpu} ({mem}GB)")
    
    return suitable_gpus

def estimate_training_time(model_params: int, batch_size: int) -> str:
    """估算训练时间"""
    # 基于经验的估算
    base_time_per_epoch = model_params / 100  # 小时
    batch_factor = 32 / batch_size  # batch size越小，时间越长
    
    time_per_epoch = base_time_per_epoch * batch_factor
    
    if time_per_epoch < 1:
        return f"{int(time_per_epoch * 60)} minutes per epoch"
    else:
        return f"{time_per_epoch:.1f} hours per epoch"
```

### 6.4.2 数据需求

```python
def estimate_data_requirements(
    task_type: str,
    target_accuracy: float,
    domain_shift: str  # 'small', 'medium', 'large'
) -> Dict:
    """
    估算数据需求
    """
    # 基础数据需求
    base_requirements = {
        "classification": {
            "small": 1000,
            "medium": 5000,
            "large": 20000
        },
        "detection": {
            "small": 5000,
            "medium": 20000,
            "large": 100000
        },
        "segmentation": {
            "small": 2000,
            "medium": 10000,
            "large": 50000
        },
        "retrieval": {
            "small": 5000,
            "medium": 20000,
            "large": 100000
        }
    }
    
    # 根据目标准确率调整
    accuracy_factor = 1.0
    if target_accuracy > 0.9:
        accuracy_factor = 2.0
    elif target_accuracy > 0.95:
        accuracy_factor = 3.0
    
    # 根据领域偏移调整
    domain_factor = {
        "small": 1.0,
        "medium": 2.0,
        "large": 5.0
    }
    
    base_count = base_requirements.get(task_type, {}).get(domain_shift, 10000)
    total_count = int(base_count * accuracy_factor * domain_factor[domain_shift])
    
    # 数据分割建议
    train_count = int(total_count * 0.7)
    val_count = int(total_count * 0.15)
    test_count = int(total_count * 0.15)
    
    # 数据增强建议
    augmentation_strategies = []
    if total_count < 5000:
        augmentation_strategies = [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "ColorJitter",
            "RandomRotation",
            "MixUp",
            "CutMix"
        ]
    elif total_count < 20000:
        augmentation_strategies = [
            "RandomResizedCrop",
            "RandomHorizontalFlip",
            "ColorJitter"
        ]
    else:
        augmentation_strategies = [
            "RandomResizedCrop",
            "RandomHorizontalFlip"
        ]
    
    return {
        "total_samples_needed": total_count,
        "data_split": {
            "train": train_count,
            "validation": val_count,
            "test": test_count
        },
        "samples_per_class": {
            "minimum": max(50, total_count // 100),
            "recommended": max(200, total_count // 20),
            "ideal": max(500, total_count // 10)
        },
        "augmentation_strategies": augmentation_strategies,
        "annotation_time_estimate": f"{total_count * 0.5 / 60:.1f} hours",
        "storage_requirement": f"{total_count * 0.5:.1f} GB"  # 假设每张图片0.5MB
    }
```

## 6.5 微调最佳实践

### 6.5.1 数据质量控制

```python
class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, clip_model, preprocess):
        self.model = clip_model
        self.preprocess = preprocess
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_dataset_quality(self, dataset_path: str) -> Dict:
        """全面检查数据集质量"""
        issues = {
            'duplicate_images': [],
            'misaligned_pairs': [],
            'low_quality_images': [],
            'invalid_captions': [],
            'class_imbalance': {}
        }
        
        # 加载数据
        with open(f"{dataset_path}/annotations.json", 'r') as f:
            annotations = json.load(f)
        
        # 检查重复图像
        image_hashes = {}
        for ann in annotations:
            img_path = f"{dataset_path}/images/{ann['image']}"
            img_hash = self._compute_image_hash(img_path)
            
            if img_hash in image_hashes:
                issues['duplicate_images'].append({
                    'original': image_hashes[img_hash],
                    'duplicate': img_path
                })
            else:
                image_hashes[img_hash] = img_path
        
        # 检查图像-文本对齐
        for ann in tqdm(annotations, desc="Checking alignment"):
            alignment_score = self._check_alignment(
                f"{dataset_path}/images/{ann['image']}", 
                ann['caption']
            )
            
            if alignment_score < 0.2:
                issues['misaligned_pairs'].append({
                    'image': ann['image'],
                    'caption': ann['caption'],
                    'score': alignment_score
                })
        
        # 检查类别平衡
        class_counts = {}
        for ann in annotations:
            label = ann['label']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        if max_count / min_count > 10:
            issues['class_imbalance'] = {
                'severe': True,
                'max_class': max(class_counts, key=class_counts.get),
                'min_class': min(class_counts, key=class_counts.get),
                'ratio': max_count / min_count
            }
        
        return issues
    
    def _check_alignment(self, image_path: str, caption: str) -> float:
        """检查图像-文本对齐度"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tensor = clip.tokenize([caption]).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).item()
        
        return similarity
    
    def _compute_image_hash(self, image_path: str) -> str:
        """计算图像哈希"""
        import hashlib
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
```

### 6.5.2 超参数优化

```python
import optuna

class CLIPHyperparameterOptimizer:
    """CLIP超参数优化器"""
    
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def objective(self, trial):
        """Optuna目标函数"""
        # 超参数搜索空间
        config = {
            'learning_rate': trial.suggest_loguniform('lr', 1e-6, 1e-3),
            'warmup_ratio': trial.suggest_uniform('warmup_ratio', 0.0, 0.2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-5, 1e-1),
            'dropout': trial.suggest_uniform('dropout', 0.0, 0.3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'unfreeze_layers': trial.suggest_int('unfreeze_layers', 1, 4),
            'temperature': trial.suggest_uniform('temperature', 0.01, 0.2)
        }
        
        # 训练模型
        fine_tuner = CLIPFineTuner()
        val_acc = fine_tuner.fine_tune(
            self.train_loader,
            self.val_loader,
            strategy="partial",
            num_epochs=5,  # 快速评估
            learning_rate=config['learning_rate'],
            warmup_steps=int(len(self.train_loader) * config['warmup_ratio']),
            use_wandb=False
        )
        
        return val_acc
    
    def optimize(self, n_trials: int = 50):
        """运行超参数优化"""
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
        )
        
        study.optimize(self.objective, n_trials=n_trials)
        
        # 返回最佳超参数
        return study.best_params
```

### 6.5.3 模型评估和部署

```python
class CLIPModelEvaluator:
    """CLIP模型评估器"""
    
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.device = next(model.parameters()).device
        
    def comprehensive_evaluation(self) -> Dict:
        """全面评估模型"""
        results = {
            'accuracy': self._compute_accuracy(),
            'retrieval_metrics': self._compute_retrieval_metrics(),
            'robustness': self._test_robustness(),
            'efficiency': self._measure_efficiency()
        }
        
        return results
    
    def _compute_accuracy(self) -> float:
        """计算分类准确率"""
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 获取预测
                logits = self.model(images)
                _, predicted = logits.max(1)
                
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def _compute_retrieval_metrics(self) -> Dict:
        """计算检索指标"""
        # 实现检索评估
        return {
            'recall@1': 0.0,
            'recall@5': 0.0,
            'recall@10': 0.0,
            'mAP': 0.0
        }
    
    def _test_robustness(self) -> Dict:
        """测试模型鲁棒性"""
        # 测试对各种扰动的鲁棒性
        return {
            'noise_robustness': 0.0,
            'blur_robustness': 0.0,
            'rotation_robustness': 0.0
        }
    
    def _measure_efficiency(self) -> Dict:
        """测量模型效率"""
        import time
        
        # 测试推理速度
        self.model.eval()
        times = []
        
        with torch.no_grad():
            for _ in range(100):
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                
                start = time.time()
                _ = self.model.encode_image(dummy_input)
                end = time.time()
                
                times.append(end - start)
        
        return {
            'avg_inference_time': np.mean(times),
            'fps': 1.0 / np.mean(times),
            'model_size_mb': self._get_model_size()
        }
    
    def _get_model_size(self) -> float:
        """获取模型大小"""
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
```

## 6.6 总结

本章详细介绍了CLIP的微调方法和资源需求：

1. **微调策略**：线性探测、部分微调、完全微调、LoRA、提示学习
2. **实战教程**：完整的数据准备、训练循环、评估流程
3. **资源分析**：硬件需求、数据需求的详细估算
4. **最佳实践**：数据质量控制、超参数优化、模型评估

掌握这些内容后，你可以根据具体需求和资源条件，选择合适的策略对CLIP进行微调。

## 下一步

继续阅读 → [第7章：进阶主题](./07-advanced-topics.md)