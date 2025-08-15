# 第5章：CLIP应用场景和玩法

## 5.1 图像检索系统

### 5.1.1 构建大规模图像检索系统

```python
import torch
import clip
import faiss
import numpy as np
from PIL import Image
import redis
import json
from pathlib import Path
from typing import List, Dict, Tuple
import hashlib

class ProductImageSearch:
    """电商产品图像搜索系统"""
    
    def __init__(self, model_name="ViT-L/14", redis_host="localhost", redis_port=6379):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # 使用FAISS构建向量索引
        self.dimension = 768  # ViT-L/14的特征维度
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 使用Redis存储元数据
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        # 产品数据
        self.product_ids = []
        
    def add_product(self, product_id: str, image_path: str, metadata: Dict):
        """添加产品到搜索系统"""
        # 提取图像特征
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy().astype('float32')
        
        # 添加到FAISS索引
        self.index.add(features)
        self.product_ids.append(product_id)
        
        # 存储元数据到Redis
        metadata['image_path'] = image_path
        self.redis_client.hset(f"product:{product_id}", mapping=metadata)
        
        # 存储图像哈希以检测重复
        image_hash = self._compute_image_hash(image_path)
        self.redis_client.set(f"hash:{image_hash}", product_id)
    
    def search_by_text(self, query: str, filters: Dict = None, k: int = 20) -> List[Dict]:
        """使用自然语言搜索产品"""
        # 编码查询文本
        text = clip.tokenize([query]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype('float32')
        
        # 搜索相似产品
        scores, indices = self.index.search(text_features, k * 2)  # 获取更多结果用于过滤
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                metadata = self.redis_client.hgetall(f"product:{product_id}")
                
                # 应用过滤器
                if filters and not self._apply_filters(metadata, filters):
                    continue
                
                result = {
                    'product_id': product_id,
                    'score': float(score),
                    'metadata': metadata
                }
                results.append(result)
                
                if len(results) >= k:
                    break
        
        return results
    
    def search_by_image(self, image_path: str, k: int = 20) -> List[Dict]:
        """使用图像搜索相似产品"""
        # 检查是否为重复图像
        image_hash = self._compute_image_hash(image_path)
        duplicate_id = self.redis_client.get(f"hash:{image_hash}")
        
        if duplicate_id:
            print(f"Found duplicate image: {duplicate_id}")
        
        # 提取图像特征
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy().astype('float32')
        
        # 搜索
        scores, indices = self.index.search(features, k)
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                metadata = self.redis_client.hgetall(f"product:{product_id}")
                
                result = {
                    'product_id': product_id,
                    'score': float(score),
                    'metadata': metadata
                }
                results.append(result)
        
        return results
    
    def hybrid_search(self, text_query: str, image_path: str, 
                     text_weight: float = 0.5, k: int = 20) -> List[Dict]:
        """混合搜索：结合文本和图像"""
        # 获取文本特征
        text = clip.tokenize([text_query]).to(self.device)
        
        # 获取图像特征
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 加权组合特征
            combined_features = (text_weight * text_features + 
                               (1 - text_weight) * image_features)
            combined_features = combined_features / combined_features.norm(dim=-1, keepdim=True)
            combined_features = combined_features.cpu().numpy().astype('float32')
        
        # 搜索
        scores, indices = self.index.search(combined_features, k)
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.product_ids):
                product_id = self.product_ids[idx]
                metadata = self.redis_client.hgetall(f"product:{product_id}")
                
                result = {
                    'product_id': product_id,
                    'score': float(score),
                    'metadata': metadata
                }
                results.append(result)
        
        return results
    
    def _compute_image_hash(self, image_path: str) -> str:
        """计算图像哈希值"""
        with open(image_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _apply_filters(self, metadata: Dict, filters: Dict) -> bool:
        """应用过滤条件"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
```

### 5.1.2 相似图像去重系统

```python
class ImageDeduplication:
    """图像去重系统"""
    
    def __init__(self, similarity_threshold=0.95):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.similarity_threshold = similarity_threshold
        self.feature_cache = {}
        
    def find_duplicates(self, image_paths: List[str]) -> List[List[str]]:
        """查找重复图像组"""
        # 提取所有图像特征
        features = []
        valid_paths = []
        
        for path in image_paths:
            if path in self.feature_cache:
                features.append(self.feature_cache[path])
                valid_paths.append(path)
            else:
                try:
                    feature = self._extract_feature(path)
                    features.append(feature)
                    valid_paths.append(path)
                    self.feature_cache[path] = feature
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    continue
        
        if not features:
            return []
        
        # 计算相似度矩阵
        features_matrix = torch.stack(features)
        similarity_matrix = features_matrix @ features_matrix.T
        
        # 查找重复组
        duplicate_groups = []
        processed = set()
        
        for i in range(len(valid_paths)):
            if i in processed:
                continue
            
            # 找到所有相似的图像
            similar_indices = torch.where(
                similarity_matrix[i] >= self.similarity_threshold
            )[0].tolist()
            
            if len(similar_indices) > 1:
                group = [valid_paths[idx] for idx in similar_indices]
                duplicate_groups.append(group)
                processed.update(similar_indices)
        
        return duplicate_groups
    
    def remove_duplicates(self, image_paths: List[str], 
                         keep_strategy: str = 'first') -> List[str]:
        """移除重复图像，返回唯一图像列表"""
        duplicate_groups = self.find_duplicates(image_paths)
        
        # 标记要保留的图像
        to_keep = set(image_paths)
        
        for group in duplicate_groups:
            if keep_strategy == 'first':
                # 保留第一个，移除其他
                to_remove = group[1:]
            elif keep_strategy == 'largest':
                # 保留最大的文件
                sizes = [(path, Path(path).stat().st_size) for path in group]
                sizes.sort(key=lambda x: x[1], reverse=True)
                to_remove = [path for path, _ in sizes[1:]]
            elif keep_strategy == 'newest':
                # 保留最新的文件
                times = [(path, Path(path).stat().st_mtime) for path in group]
                times.sort(key=lambda x: x[1], reverse=True)
                to_remove = [path for path, _ in times[1:]]
            else:
                raise ValueError(f"Unknown strategy: {keep_strategy}")
            
            to_keep -= set(to_remove)
        
        return list(to_keep)
    
    def _extract_feature(self, image_path: str) -> torch.Tensor:
        """提取图像特征"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        
        return features.squeeze(0)
```

## 5.2 内容理解和生成

### 5.2.1 智能图像标注系统

```python
class SmartImageTagger:
    """智能图像标注系统"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
        # 多层次标签体系
        self.tag_hierarchy = {
            'style': ['photograph', 'illustration', 'painting', 'sketch', '3D render', 
                     'cartoon', 'anime', 'realistic', 'abstract'],
            'color': ['black and white', 'colorful', 'monochrome', 'vibrant', 
                     'dark', 'bright', 'pastel', 'neon'],
            'composition': ['close-up', 'wide shot', 'portrait', 'landscape', 
                          'aerial view', 'macro', 'panoramic'],
            'mood': ['happy', 'sad', 'peaceful', 'energetic', 'mysterious', 
                    'romantic', 'dramatic', 'serene'],
            'time': ['daytime', 'night', 'sunrise', 'sunset', 'golden hour', 
                    'blue hour', 'twilight'],
            'weather': ['sunny', 'cloudy', 'rainy', 'snowy', 'foggy', 'stormy'],
            'subject': ['person', 'animal', 'nature', 'urban', 'architecture', 
                       'food', 'technology', 'art', 'sports']
        }
        
    def generate_tags(self, image_path: str, threshold: float = 0.2) -> Dict[str, List[Tuple[str, float]]]:
        """为图像生成多层次标签"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        results = {}
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for category, tags in self.tag_hierarchy.items():
                # 为每个类别创建文本提示
                prompts = [f"a {tag} image" if category != 'subject' 
                          else f"an image of {tag}" for tag in tags]
                
                text = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 计算相似度
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
                # 收集超过阈值的标签
                category_tags = []
                for i, tag in enumerate(tags):
                    score = similarities[0][i].item()
                    if score >= threshold:
                        category_tags.append((tag, score))
                
                # 按分数排序
                category_tags.sort(key=lambda x: x[1], reverse=True)
                results[category] = category_tags
        
        return results
    
    def generate_description(self, image_path: str) -> str:
        """生成图像的自然语言描述"""
        tags = self.generate_tags(image_path)
        
        description_parts = []
        
        # 主题
        if tags.get('subject'):
            subject = tags['subject'][0][0]
            description_parts.append(f"This image shows {subject}")
        
        # 风格
        if tags.get('style'):
            style = tags['style'][0][0]
            description_parts.append(f"in {style} style")
        
        # 颜色
        if tags.get('color'):
            color = tags['color'][0][0]
            description_parts.append(f"with {color} colors")
        
        # 情绪
        if tags.get('mood'):
            mood = tags['mood'][0][0]
            description_parts.append(f"creating a {mood} mood")
        
        # 组合描述
        if description_parts:
            description = " ".join(description_parts) + "."
            return description.capitalize()
        else:
            return "An image."
    
    def suggest_keywords(self, image_path: str, num_keywords: int = 10) -> List[str]:
        """为SEO或搜索优化建议关键词"""
        tags = self.generate_tags(image_path, threshold=0.1)
        
        # 收集所有标签及其分数
        all_tags = []
        for category, category_tags in tags.items():
            for tag, score in category_tags:
                # 根据类别调整权重
                weight = score
                if category == 'subject':
                    weight *= 1.5  # 主题更重要
                elif category == 'style':
                    weight *= 1.2
                
                all_tags.append((tag, weight))
        
        # 排序并返回前N个
        all_tags.sort(key=lambda x: x[1], reverse=True)
        keywords = [tag for tag, _ in all_tags[:num_keywords]]
        
        return keywords
```

### 5.2.2 创意内容生成助手

```python
class CreativeAssistant:
    """创意内容生成助手"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        
    def find_metaphors(self, image_path: str) -> List[Dict]:
        """为图像找到视觉隐喻"""
        metaphors = [
            {'concept': 'freedom', 'visual': 'bird flying', 'strength': 0},
            {'concept': 'growth', 'visual': 'tree growing', 'strength': 0},
            {'concept': 'journey', 'visual': 'winding road', 'strength': 0},
            {'concept': 'strength', 'visual': 'mountain', 'strength': 0},
            {'concept': 'peace', 'visual': 'calm water', 'strength': 0},
            {'concept': 'innovation', 'visual': 'light bulb', 'strength': 0},
            {'concept': 'connection', 'visual': 'bridge', 'strength': 0},
            {'concept': 'transformation', 'visual': 'butterfly', 'strength': 0}
        ]
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 测试每个隐喻
            for metaphor in metaphors:
                text = clip.tokenize([f"an image representing {metaphor['concept']} through {metaphor['visual']}"]).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarity = (image_features @ text_features.T).item()
                metaphor['strength'] = similarity
        
        # 排序并返回相关的隐喻
        metaphors.sort(key=lambda x: x['strength'], reverse=True)
        return [m for m in metaphors if m['strength'] > 0.2]
    
    def suggest_color_palette(self, image_path: str) -> Dict:
        """建议配色方案"""
        color_moods = {
            'warm': ['red', 'orange', 'yellow', 'brown'],
            'cool': ['blue', 'green', 'purple', 'cyan'],
            'neutral': ['gray', 'beige', 'white', 'black'],
            'vibrant': ['bright red', 'electric blue', 'neon green', 'hot pink'],
            'pastel': ['soft pink', 'baby blue', 'mint green', 'lavender'],
            'earth': ['brown', 'green', 'rust', 'sand']
        }
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        palette_scores = {}
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for palette_name, colors in color_moods.items():
                # 测试配色方案
                prompts = [f"an image with {color} colors" for color in colors]
                text = clip.tokenize(prompts).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = image_features @ text_features.T
                avg_similarity = similarities.mean().item()
                palette_scores[palette_name] = avg_similarity
        
        # 找到最佳配色方案
        best_palette = max(palette_scores, key=palette_scores.get)
        
        return {
            'recommended_palette': best_palette,
            'colors': color_moods[best_palette],
            'confidence': palette_scores[best_palette],
            'all_scores': palette_scores
        }
    
    def generate_social_media_captions(self, image_path: str, platform: str = 'instagram') -> List[str]:
        """生成社交媒体标题"""
        # 分析图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 不同平台的风格
        platform_styles = {
            'instagram': {
                'tone': ['inspirational', 'aesthetic', 'lifestyle'],
                'length': 'medium',
                'hashtags': True,
                'emoji': True
            },
            'twitter': {
                'tone': ['witty', 'informative', 'engaging'],
                'length': 'short',
                'hashtags': True,
                'emoji': False
            },
            'linkedin': {
                'tone': ['professional', 'insightful', 'thoughtful'],
                'length': 'long',
                'hashtags': False,
                'emoji': False
            }
        }
        
        style = platform_styles.get(platform, platform_styles['instagram'])
        
        # 检测图像内容
        content_types = ['nature', 'urban', 'people', 'food', 'art', 'technology']
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 检测内容类型
            prompts = [f"a photo of {content}" for content in content_types]
            text = clip.tokenize(prompts).to(self.device)
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarities = (image_features @ text_features.T).squeeze()
            best_content_idx = similarities.argmax().item()
            content_type = content_types[best_content_idx]
        
        # 生成标题模板
        captions = []
        
        if content_type == 'nature':
            captions = [
                "Finding peace in nature's embrace 🌿",
                "Mother Earth showing off again",
                "This view though... #NatureLover"
            ]
        elif content_type == 'urban':
            captions = [
                "City lights and urban nights ✨",
                "Concrete jungle vibes",
                "Where dreams meet skylines #CityLife"
            ]
        elif content_type == 'food':
            captions = [
                "Good food = Good mood 🍽️",
                "Feast your eyes on this!",
                "Foodie paradise found #FoodStagram"
            ]
        
        # 根据平台调整
        if not style['emoji']:
            captions = [cap.replace('🌿', '').replace('✨', '').replace('🍽️', '') for cap in captions]
        if not style['hashtags']:
            captions = [cap.split('#')[0].strip() for cap in captions]
        
        return captions
```

## 5.3 多模态交互应用

### 5.3.1 视觉对话系统

```python
class VisualDialogSystem:
    """视觉对话系统"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
        self.conversation_history = []
        self.image_context = None
        
    def set_image_context(self, image_path: str):
        """设置对话的图像上下文"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            self.image_context = self.model.encode_image(image_tensor)
            self.image_context = self.image_context / self.image_context.norm(dim=-1, keepdim=True)
        
        # 初始分析
        self.image_analysis = self._analyze_image()
        self.conversation_history = []
        
    def _analyze_image(self) -> Dict:
        """分析图像基本信息"""
        if self.image_context is None:
            return {}
        
        analyses = {}
        
        # 检测对象
        objects = ['person', 'animal', 'vehicle', 'building', 'plant', 'food', 'furniture']
        object_prompts = [f"a photo with a {obj}" for obj in objects]
        
        # 检测场景
        scenes = ['indoor', 'outdoor', 'street', 'nature', 'beach', 'mountain', 'city']
        scene_prompts = [f"a photo taken {scene}" if scene in ['indoor', 'outdoor'] 
                        else f"a photo of a {scene}" for scene in scenes]
        
        with torch.no_grad():
            # 对象检测
            text = clip.tokenize(object_prompts).to(self.device)
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            object_similarities = (self.image_context @ text_features.T).squeeze()
            detected_objects = [obj for obj, sim in zip(objects, object_similarities) if sim > 0.25]
            analyses['objects'] = detected_objects
            
            # 场景检测
            text = clip.tokenize(scene_prompts).to(self.device)
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            scene_similarities = (self.image_context @ text_features.T).squeeze()
            best_scene_idx = scene_similarities.argmax().item()
            analyses['scene'] = scenes[best_scene_idx]
        
        return analyses
    
    def answer_question(self, question: str) -> str:
        """回答关于图像的问题"""
        if self.image_context is None:
            return "Please set an image context first."
        
        # 记录问题
        self.conversation_history.append({'type': 'question', 'content': question})
        
        # 问题类型识别
        question_lower = question.lower()
        
        # 计数问题
        if 'how many' in question_lower:
            return self._answer_counting_question(question)
        
        # 是否问题
        elif question_lower.startswith(('is', 'are', 'does', 'do', 'can')):
            return self._answer_yes_no_question(question)
        
        # 什么问题
        elif 'what' in question_lower:
            return self._answer_what_question(question)
        
        # 哪里问题
        elif 'where' in question_lower:
            return self._answer_where_question(question)
        
        # 描述问题
        else:
            return self._answer_descriptive_question(question)
    
    def _answer_yes_no_question(self, question: str) -> str:
        """回答是/否问题"""
        # 创建肯定和否定的陈述
        affirmative = question.replace('?', '').replace('Is', 'This is').replace('Are', 'These are')
        negative = question.replace('?', '').replace('Is', 'This is not').replace('Are', 'These are not')
        
        prompts = [affirmative, negative]
        text = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarities = (self.image_context @ text_features.T).squeeze()
            
        if similarities[0] > similarities[1]:
            answer = "Yes"
        else:
            answer = "No"
        
        # 添加置信度
        confidence = abs(similarities[0] - similarities[1]).item()
        if confidence < 0.1:
            answer += ", but I'm not very certain"
        elif confidence > 0.3:
            answer += ", definitely"
        
        self.conversation_history.append({'type': 'answer', 'content': answer})
        return answer
    
    def _answer_counting_question(self, question: str) -> str:
        """回答计数问题"""
        # 提取要计数的对象
        import re
        match = re.search(r'how many (\w+)', question.lower())
        if not match:
            return "I couldn't understand what to count."
        
        item = match.group(1)
        
        # 测试不同数量
        numbers = ['no', 'one', 'two', 'three', 'four', 'five', 'many']
        prompts = [f"a photo with {num} {item}" for num in numbers]
        
        text = clip.tokenize(prompts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarities = (self.image_context @ text_features.T).squeeze()
            best_idx = similarities.argmax().item()
        
        answer = f"I can see {numbers[best_idx]} {item} in the image"
        
        self.conversation_history.append({'type': 'answer', 'content': answer})
        return answer
    
    def _answer_what_question(self, question: str) -> str:
        """回答'什么'类问题"""
        if 'color' in question.lower():
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'brown', 'gray']
            prompts = [f"a predominantly {color} image" for color in colors]
            
            text = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (self.image_context @ text_features.T).squeeze()
                best_idx = similarities.argmax().item()
            
            answer = f"The dominant color appears to be {colors[best_idx]}"
        
        elif 'doing' in question.lower():
            actions = ['standing', 'sitting', 'walking', 'running', 'eating', 'working', 'playing', 'talking']
            prompts = [f"a photo of someone {action}" for action in actions]
            
            text = clip.tokenize(prompts).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (self.image_context @ text_features.T).squeeze()
                best_idx = similarities.argmax().item()
            
            answer = f"It appears someone is {actions[best_idx]}"
        
        else:
            # 通用what问题
            answer = f"Based on my analysis, the image shows {', '.join(self.image_analysis.get('objects', ['something']))}"
        
        self.conversation_history.append({'type': 'answer', 'content': answer})
        return answer
    
    def _answer_where_question(self, question: str) -> str:
        """回答'哪里'类问题"""
        scene = self.image_analysis.get('scene', 'unknown location')
        answer = f"This appears to be {scene}"
        
        self.conversation_history.append({'type': 'answer', 'content': answer})
        return answer
    
    def _answer_descriptive_question(self, question: str) -> str:
        """回答描述性问题"""
        # 基于图像分析生成描述
        objects = self.image_analysis.get('objects', [])
        scene = self.image_analysis.get('scene', 'location')
        
        if objects:
            answer = f"I can see {', '.join(objects)} in this {scene} setting"
        else:
            answer = f"This appears to be a {scene} scene"
        
        self.conversation_history.append({'type': 'answer', 'content': answer})
        return answer
```

### 5.3.2 增强现实（AR）应用

```python
class CLIPARAssistant:
    """基于CLIP的AR助手"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device)
        self.object_database = {}
        
    def register_ar_object(self, object_id: str, trigger_descriptions: List[str], 
                          ar_content: Dict):
        """注册AR对象和触发条件"""
        # 预计算触发描述的特征
        text = clip.tokenize(trigger_descriptions).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            # 平均多个描述
            trigger_feature = text_features.mean(dim=0)
            trigger_feature = trigger_feature / trigger_feature.norm()
        
        self.object_database[object_id] = {
            'trigger_feature': trigger_feature,
            'descriptions': trigger_descriptions,
            'ar_content': ar_content,
            'threshold': 0.3
        }
    
    def detect_ar_triggers(self, frame: np.ndarray) -> List[Dict]:
        """检测当前帧中的AR触发器"""
        # 将numpy数组转换为PIL图像
        image = Image.fromarray(frame)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        triggered_objects = []
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for object_id, obj_data in self.object_database.items():
                similarity = (image_features @ obj_data['trigger_feature']).item()
                
                if similarity >= obj_data['threshold']:
                    triggered_objects.append({
                        'object_id': object_id,
                        'confidence': similarity,
                        'ar_content': obj_data['ar_content']
                    })
        
        return triggered_objects
    
    def contextual_ar_suggestions(self, frame: np.ndarray) -> List[Dict]:
        """基于场景理解的AR建议"""
        image = Image.fromarray(frame)
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        suggestions = []
        
        # 场景类型检测
        scene_types = {
            'restaurant': {
                'prompts': ['a photo of a restaurant', 'dining table with food'],
                'ar_suggestions': ['menu overlay', 'nutritional info', 'reviews']
            },
            'museum': {
                'prompts': ['a photo of a museum', 'artwork on display'],
                'ar_suggestions': ['artwork info', 'audio guide', 'artist biography']
            },
            'street': {
                'prompts': ['a street view', 'urban environment'],
                'ar_suggestions': ['navigation arrows', 'store info', 'traffic updates']
            },
            'nature': {
                'prompts': ['nature scenery', 'outdoor landscape'],
                'ar_suggestions': ['species identification', 'trail maps', 'weather info']
            }
        }
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            for scene_name, scene_data in scene_types.items():
                text = clip.tokenize(scene_data['prompts']).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).squeeze()
                max_similarity = similarities.max().item()
                
                if max_similarity > 0.25:
                    suggestions.append({
                        'scene_type': scene_name,
                        'confidence': max_similarity,
                        'ar_suggestions': scene_data['ar_suggestions']
                    })
        
        return suggestions
```

## 5.4 创新应用案例

### 5.4.1 智能相册管理

```python
class SmartPhotoAlbum:
    """智能相册管理系统"""
    
    def __init__(self, album_path: str):
        self.album_path = Path(album_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.photo_index = {}
        
    def organize_by_events(self):
        """根据事件自动组织照片"""
        event_patterns = {
            'wedding': ['wedding dress', 'bride and groom', 'wedding ceremony'],
            'birthday': ['birthday cake', 'candles', 'party', 'celebration'],
            'vacation': ['beach', 'mountain', 'tourist', 'landmark'],
            'graduation': ['graduation cap', 'diploma', 'ceremony'],
            'sports': ['playing sports', 'stadium', 'athletic'],
            'concert': ['stage', 'crowd', 'performance', 'music'],
            'family': ['family gathering', 'group photo', 'home']
        }
        
        photo_events = {}
        
        for photo_path in self.album_path.glob("**/*.jpg"):
            image = Image.open(photo_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                best_event = None
                best_score = 0
                
                for event_name, patterns in event_patterns.items():
                    prompts = [f"a photo of {pattern}" for pattern in patterns]
                    text = clip.tokenize(prompts).to(self.device)
                    text_features = self.model.encode_text(text)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    similarities = (image_features @ text_features.T).squeeze()
                    avg_similarity = similarities.mean().item()
                    
                    if avg_similarity > best_score:
                        best_score = avg_similarity
                        best_event = event_name
                
                if best_score > 0.25:
                    if best_event not in photo_events:
                        photo_events[best_event] = []
                    photo_events[best_event].append(str(photo_path))
        
        # 创建事件文件夹
        for event_name, photos in photo_events.items():
            event_folder = self.album_path / f"Events/{event_name}"
            event_folder.mkdir(parents=True, exist_ok=True)
            
            for photo in photos:
                # 创建符号链接或复制文件
                dest = event_folder / Path(photo).name
                if not dest.exists():
                    dest.symlink_to(photo)
        
        return photo_events
    
    def create_story_timeline(self) -> List[Dict]:
        """创建照片故事时间线"""
        from datetime import datetime
        import exifread
        
        timeline = []
        
        for photo_path in self.album_path.glob("**/*.jpg"):
            # 获取拍摄时间
            with open(photo_path, 'rb') as f:
                tags = exifread.process_file(f)
                date_taken = tags.get('EXIF DateTimeOriginal')
                
                if date_taken:
                    date_taken = datetime.strptime(str(date_taken), '%Y:%m:%d %H:%M:%S')
                else:
                    # 使用文件修改时间
                    date_taken = datetime.fromtimestamp(photo_path.stat().st_mtime)
            
            # 分析照片内容
            image = Image.open(photo_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 生成描述
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # 检测主要内容
                content_prompts = [
                    "a photo of people",
                    "a landscape photo",
                    "a photo of food",
                    "an indoor photo",
                    "an outdoor photo"
                ]
                
                text = clip.tokenize(content_prompts).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).squeeze()
                best_idx = similarities.argmax().item()
                description = content_prompts[best_idx].replace("a photo of ", "")
            
            timeline.append({
                'path': str(photo_path),
                'date': date_taken,
                'description': description,
                'year': date_taken.year,
                'month': date_taken.strftime('%B')
            })
        
        # 按时间排序
        timeline.sort(key=lambda x: x['date'])
        
        return timeline
    
    def find_best_shots(self, num_photos: int = 10) -> List[str]:
        """找出最佳照片"""
        quality_criteria = [
            "a high quality professional photo",
            "a well-composed photograph",
            "a sharp, in-focus image",
            "good lighting and exposure",
            "aesthetically pleasing composition"
        ]
        
        photo_scores = []
        
        for photo_path in self.album_path.glob("**/*.jpg"):
            image = Image.open(photo_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                text = clip.tokenize(quality_criteria).to(self.device)
                text_features = self.model.encode_text(text)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = (image_features @ text_features.T).squeeze()
                quality_score = similarities.mean().item()
                
                photo_scores.append((str(photo_path), quality_score))
        
        # 排序并返回最佳照片
        photo_scores.sort(key=lambda x: x[1], reverse=True)
        best_photos = [path for path, _ in photo_scores[:num_photos]]
        
        return best_photos
```

## 5.5 总结

本章介绍了CLIP的各种实际应用和创新玩法，包括：

1. **图像检索系统**：构建大规模搜索引擎、去重系统
2. **内容理解生成**：智能标注、创意辅助
3. **多模态交互**：视觉对话、AR应用
4. **创新应用**：智能相册、故事生成

这些应用展示了CLIP在实际场景中的强大能力和灵活性。

## 下一步

继续阅读 → [第6章：CLIP微调教程](./06-fine-tuning.md)