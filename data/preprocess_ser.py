#!/usr/bin/env python3

import os
import json
import random
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# 固定随机种子
RANDOM_SEED = 42

# 固定窗口大小
HISTORY_LEN = 3  # 每个样本的历史长度
SAMPLES_PER_USER = 3  # 每个用户生成的样本数

# 数据集路径
BASE_PATH = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/SER_Dataset"
IMAGES_ROOT = os.path.join(BASE_PATH, "Images")
ANNOTATIONS_PATH = os.path.join(BASE_PATH, "Annotations", "all_annos.json")
CAPTIONS_PATH = os.path.join(BASE_PATH, "ser30k_captions.json")

# 划分比例
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

# 输出路径
OUTPUT_DIR = os.path.join(BASE_PATH, "processed_dataset")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")


def load_captions():
    """加载描述数据"""
    print("=" * 80)
    print("Loading captions...")
    print("=" * 80)
    
    captions = {}
    if os.path.exists(CAPTIONS_PATH):
        with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
            captions = json.load(f)
        print(f"  - Loaded {len(captions):,} captions")
    else:
        print(f"  - Warning: {CAPTIONS_PATH} not found")
    
    return captions


def find_image_path(topic: str, filename: str) -> str:
    """查找图片路径（尝试更多扩展名）"""
    topic_dir = os.path.join(IMAGES_ROOT, topic)
    if not os.path.exists(topic_dir):
        return None
    
    # 如果文件名已有扩展名，直接检查
    full_path = os.path.join(topic_dir, filename)
    if os.path.exists(full_path):
        return full_path
    
    # 尝试更多图片扩展名
    extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.TIF',
        '.Jpg', '.Jpeg', '.Png', '.Gif', '.Bmp', '.Webp', '.Tiff', '.Tif'
    ]
    
    # 去掉可能存在的扩展名
    base_name = os.path.splitext(filename)[0]
    
    for ext in extensions:
        image_path = os.path.join(topic_dir, base_name + ext)
        if os.path.exists(image_path):
            return image_path
    
    # 尝试原文件名加扩展名
    for ext in extensions:
        image_path = os.path.join(topic_dir, filename + ext)
        if os.path.exists(image_path):
            return image_path
    
    return None


def get_topic_to_images_mapping():
    """获取每个topic文件夹下的所有图像文件"""
    print("\n" + "=" * 80)
    print("Scanning Images directory...")
    print("=" * 80)
    
    topic_to_images = defaultdict(list)
    
    if not os.path.exists(IMAGES_ROOT):
        raise FileNotFoundError(f"Images directory not found: {IMAGES_ROOT}")
    
    # 遍历所有topic文件夹
    topics = [d for d in os.listdir(IMAGES_ROOT) 
              if os.path.isdir(os.path.join(IMAGES_ROOT, d)) and not d.startswith('.')]
    
    print(f"  - Found {len(topics)} topic folders")
    
    for topic in topics:
        topic_path = os.path.join(IMAGES_ROOT, topic)
        # 获取所有图像文件
        image_files = [f for f in os.listdir(topic_path) 
                      if not f.startswith('.') and 
                      f.lower().endswith(('.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'))]
        
        if len(image_files) > 0:
            topic_to_images[topic] = sorted(image_files)  # 排序保证一致性
    
    print(f"  - Topics with images: {len(topic_to_images)}")
    
    # 统计信息
    total_images = sum(len(images) for images in topic_to_images.values())
    print(f"  - Total images: {total_images:,}")
    
    # 显示一些统计
    image_counts = [len(images) for images in topic_to_images.values()]
    if image_counts:
        print(f"  - Min images per topic: {min(image_counts)}")
        print(f"  - Max images per topic: {max(image_counts)}")
        print(f"  - Avg images per topic: {sum(image_counts) / len(image_counts):.1f}")
    
    # 过滤：至少有 HISTORY_LEN 张图片的 topic
    filtered_topics = {topic: images for topic, images in topic_to_images.items() 
                       if len(images) >= HISTORY_LEN}
    
    print(f"  - Topics with >= {HISTORY_LEN} images: {len(filtered_topics)}")
    print(f"  - Filtered out: {len(topic_to_images) - len(filtered_topics)} topics")
    
    return filtered_topics


def create_dataset_samples(topic_to_images: Dict[str, List[str]], 
                           captions: Dict[str, str]) -> List[Dict]:
    """
    创建数据集样本（固定窗口：history_len=3, target=1）
    
    对于每个topic（作为用户）：
    - 从该topic中随机抽取3张图片作为历史
    - 从其他topic中随机抽取1张图片作为target
    - 每个用户生成3个样本（即抽取3次不同的target）
    - 只保留所有图片（历史3个+目标1个）都存在的样本
    """
    print("\n" + "=" * 80)
    print("Creating dataset samples...")
    print("=" * 80)
    print(f"  - Window size: history={HISTORY_LEN}, target=1")
    print(f"  - Samples per user: {SAMPLES_PER_USER}")
    print(f"  - Filter: Only keeping samples with all 4 images available")
    
    # 设置随机种子
    random.seed(RANDOM_SEED)
    
    # 获取所有topic列表
    all_topics = sorted(list(topic_to_images.keys()))
    print(f"  - Total topics (users): {len(all_topics)}")
    
    samples = []
    users_with_samples = 0
    total_potential_samples = 0
    filtered_by_insufficient_images = 0
    filtered_by_missing_files = 0
    
    for user_topic in all_topics:
        # 该用户的所有图片
        all_images = topic_to_images[user_topic]
        
        # 需要至少 HISTORY_LEN 张图片
        if len(all_images) < HISTORY_LEN:
            filtered_by_insufficient_images += 1
            continue
        
        # 获取其他 topics（用于抽取 target）
        other_topics = [t for t in all_topics if t != user_topic]
        
        if len(other_topics) == 0:
            continue
        
        user_has_samples = False
        
        # 为该用户生成 SAMPLES_PER_USER 个样本
        for sample_idx in range(SAMPLES_PER_USER):
            total_potential_samples += 1
            
            # 1. 从该用户的 topic 中随机抽取 HISTORY_LEN 张图片作为历史
            history_filenames = random.sample(all_images, HISTORY_LEN)
            
            # 验证历史图片是否都存在
            history_image_paths = []
            all_history_images_exist = True
            for filename in history_filenames:
                img_path = find_image_path(user_topic, filename)
                if img_path is None:
                    all_history_images_exist = False
                    break
                history_image_paths.append(img_path)
            
            if not all_history_images_exist:
                filtered_by_missing_files += 1
                continue
            
            # 2. 从其他 topic 中随机抽取一个作为 target
            target_topic = random.choice(other_topics)
            target_images = topic_to_images[target_topic]
            
            if len(target_images) == 0:
                filtered_by_missing_files += 1
                continue
            
            target_filename = random.choice(target_images)
            
            # 验证目标图片是否存在
            target_image_path = find_image_path(target_topic, target_filename)
            if target_image_path is None:
                filtered_by_missing_files += 1
                continue
            
            # === 所有图片都存在，构建样本 ===
            # 构建历史物品 ID 和信息
            history_item_ids = [f"{user_topic}/{fn}" for fn in history_filenames]
            history_items_info = []
            
            for idx, filename in enumerate(history_filenames):
                item_id = f"{user_topic}/{filename}"
                caption = captions.get(item_id, "")
                
                history_items_info.append({
                    'item_id': item_id,
                    'caption': caption,
                    'image_path': history_image_paths[idx]
                })
            
            # 构建目标物品信息
            target_item_id = f"{target_topic}/{target_filename}"
            target_caption = captions.get(target_item_id, "")
            target_item_info = {
                'item_id': target_item_id,
                'caption': target_caption,
                'image_path': target_image_path
            }
            
            sample = {
                'user_id': user_topic,  # topic 作为 user_id
                'history_item_ids': history_item_ids,
                'history_items_info': history_items_info,
                'target_item_id': target_item_id,
                'target_item_info': target_item_info,
                'num_interactions': HISTORY_LEN,
                'sample_index': sample_idx  # 该用户的第几个样本
            }
            
            samples.append(sample)
            user_has_samples = True
        
        if user_has_samples:
            users_with_samples += 1
    
    print(f"  - Total potential samples: {total_potential_samples:,}")
    print(f"  - Filtered by insufficient images: {filtered_by_insufficient_images:,}")
    print(f"  - Filtered by missing files: {filtered_by_missing_files:,} ({filtered_by_missing_files/total_potential_samples*100:.1f}%)")
    print(f"  - Created {len(samples):,} valid samples from {users_with_samples:,} users")
    if users_with_samples > 0:
        print(f"  - Average samples per user: {len(samples)/users_with_samples:.1f}")
    
    # 统计样本数分布
    samples_per_user = Counter(s['user_id'] for s in samples)
    sample_counts = Counter(samples_per_user.values())
    print(f"\n  - Sample distribution:")
    for count in sorted(sample_counts.keys())[:10]:
        print(f"      {count} sample(s): {sample_counts[count]:,} users")
    if len(sample_counts) > 10:
        print(f"      ... (showing first 10)")
    
    # 统计有caption的样本
    samples_with_history_captions = sum(1 for s in samples 
                                         if any(item.get('caption') for item in s['history_items_info']))
    samples_with_target_captions = sum(1 for s in samples if s['target_item_info'].get('caption'))
    
    print(f"\n  - Data coverage (all samples have images):")
    print(f"      History captions: {samples_with_history_captions:,}/{len(samples):,} ({samples_with_history_captions/len(samples)*100:.1f}%)")
    print(f"      Target captions: {samples_with_target_captions:,}/{len(samples):,} ({samples_with_target_captions/len(samples)*100:.1f}%)")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集为train/val/test（固定val为15，test按比例）"""
    print("\n" + "=" * 80)
    print("Splitting dataset...")
    print("=" * 80)
    
    # 设置随机种子保证划分一致性
    random.seed(RANDOM_SEED)
    
    # 按用户划分，确保同一用户的所有样本在同一集合中
    user_to_samples = defaultdict(list)
    for sample in samples:
        user_to_samples[sample['user_id']].append(sample)
    
    user_ids = sorted(list(user_to_samples.keys()))
    random.shuffle(user_ids)
    
    # 固定验证集为15个样本
    VAL_SIZE = 15
    
    val_samples = []
    remaining_user_ids = []
    
    # 从每个用户抽取样本直到达到15个
    for uid in user_ids:
        user_samples = user_to_samples[uid]
        if len(val_samples) < VAL_SIZE:
            # 计算还需要多少样本
            needed = VAL_SIZE - len(val_samples)
            # 从该用户的样本中抽取（不超过需要的数量）
            to_add = min(needed, len(user_samples))
            val_samples.extend(user_samples[:to_add])
            
            # 如果该用户还有剩余样本，加入到remaining
            if to_add < len(user_samples):
                remaining_user_ids.append(uid)
        else:
            remaining_user_ids.append(uid)
    
    # 剩余用户按 train/test 划分
    total_remaining = len(remaining_user_ids)
    if total_remaining > 0:
        # 根据原始比例计算 train 占剩余的比例
        train_ratio_of_remaining = TRAIN_RATIO / (TRAIN_RATIO + TEST_RATIO)
        train_user_count = int(total_remaining * train_ratio_of_remaining)
        
        train_user_ids = set(remaining_user_ids[:train_user_count])
        test_user_ids = set(remaining_user_ids[train_user_count:])
    else:
        train_user_ids = set()
        test_user_ids = set()
    
    # 收集训练集和测试集样本
    train_samples = []
    test_samples = []
    
    for uid in train_user_ids:
        train_samples.extend(user_to_samples[uid])
    
    for uid in test_user_ids:
        test_samples.extend(user_to_samples[uid])
    
    # 统计信息
    total_users = len(user_ids)
    total_samples = len(samples)
    
    print(f"  - Total users: {total_users}")
    print(f"  - Val users: {len([uid for uid in user_ids if any(s['user_id'] == uid for s in val_samples)])}")
    print(f"  - Train users: {len(train_user_ids)} ({len(train_user_ids)/total_users*100:.1f}%)")
    print(f"  - Test users: {len(test_user_ids)} ({len(test_user_ids)/total_users*100:.1f}%)")
    
    print(f"\n  - Total samples: {total_samples:,}")
    print(f"  - Val samples: {len(val_samples):,} (fixed at {VAL_SIZE})")
    print(f"  - Train samples: {len(train_samples):,} ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  - Test samples: {len(test_samples):,} ({len(test_samples)/total_samples*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def save_datasets(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """保存数据集到JSON文件"""
    print("\n" + "=" * 80)
    print("Saving datasets...")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 保存训练集
    with open(OUTPUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved train set: {OUTPUT_TRAIN} ({len(train_samples):,} samples)")
    
    # 保存验证集
    with open(OUTPUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved val set: {OUTPUT_VAL} ({len(val_samples):,} samples)")
    
    # 保存测试集
    with open(OUTPUT_TEST, 'w', encoding='utf-8') as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=2)
    print(f"  - Saved test set: {OUTPUT_TEST} ({len(test_samples):,} samples)")


def print_statistics(train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    all_samples = train_samples + val_samples + test_samples
    
    print(f"\n[1] Sample Counts:")
    print(f"  - Train: {len(train_samples):,}")
    print(f"  - Val: {len(val_samples):,}")
    print(f"  - Test: {len(test_samples):,}")
    print(f"  - Total: {len(all_samples):,}")
    
    print(f"\n[2] Configuration:")
    print(f"  - History length: {HISTORY_LEN} (fixed)")
    print(f"  - Target: 1 item (fixed)")
    print(f"  - Samples per user: {SAMPLES_PER_USER}")
    print(f"  - Image filter: All 4 images must exist")
    
    print(f"\n[3] Sample Data Structure (example):")
    if all_samples:
        sample = all_samples[0]
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - user_id: {sample['user_id']}")
        print(f"  - num_interactions: {sample['num_interactions']} (always {HISTORY_LEN})")
        print(f"  - sample_index: {sample.get('sample_index', 'N/A')}")
        print(f"  - History items (always 3, all have images):")
        for i, item in enumerate(sample['history_items_info']):
            print(f"      [{i+1}] item_id: {item['item_id']}, "
                  f"has_caption: {bool(item.get('caption'))}, "
                  f"image: ...{item.get('image_path', 'N/A')[-40:]}")
        print(f"  - Target item (has image):")
        target = sample['target_item_info']
        print(f"      item_id: {target['item_id']}, "
              f"has_caption: {bool(target.get('caption'))}, "
              f"image: ...{target.get('image_path', 'N/A')[-40:]}")
    
    print(f"\n[4] User Distribution:")
    user_sample_counts = {}
    for s in all_samples:
        uid = s['user_id']
        user_sample_counts[uid] = user_sample_counts.get(uid, 0) + 1
    
    if user_sample_counts:
        avg_samples_per_user = sum(user_sample_counts.values()) / len(user_sample_counts)
        print(f"  - Total unique users: {len(user_sample_counts):,}")
        print(f"  - Average samples per user: {avg_samples_per_user:.1f}")
        print(f"  - Users with {SAMPLES_PER_USER} samples: {sum(1 for c in user_sample_counts.values() if c == SAMPLES_PER_USER):,}")
        print(f"  - Users with < {SAMPLES_PER_USER} samples: {sum(1 for c in user_sample_counts.values() if c < SAMPLES_PER_USER):,}")


def main():
    """主函数"""
    print("=" * 80)
    print("SER Dataset Processing (Fixed Window)")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Window: history={HISTORY_LEN}, target=1")
    print(f"Samples per user: {SAMPLES_PER_USER}")
    print(f"Image filter: All 4 images must exist")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    
    # 1. 加载captions
    captions = load_captions()
    
    # 2. 获取topic到图像的映射
    topic_to_images = get_topic_to_images_mapping()
    
    # 3. 创建数据集样本（固定窗口，过滤缺失图片）
    samples = create_dataset_samples(topic_to_images, captions)
    
    # 4. 划分数据集
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    # 5. 保存数据集
    save_datasets(train_samples, val_samples, test_samples)
    
    # 6. 打印统计信息
    print_statistics(train_samples, val_samples, test_samples)
    
    print("\n" + "=" * 80)
    print("Dataset processing completed!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {OUTPUT_TRAIN}")
    print(f"  - {OUTPUT_VAL}")
    print(f"  - {OUTPUT_TEST}")


if __name__ == "__main__":
    main()