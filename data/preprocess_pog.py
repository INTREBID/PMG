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
MIN_SEQUENCE_LEN = HISTORY_LEN + 1  # 最少需要的item数量（3个历史+1个target=4）

# 数据集路径
POG_BASE_DIR = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/POG"
USER_DATA_FILE = os.path.join(POG_BASE_DIR, "subset_user_2000.txt")
CAPTIONS_FILE = os.path.join(POG_BASE_DIR, "captions_sampled.json")
USER_STYLES_FILE = os.path.join(POG_BASE_DIR, "user_styles.json")
IMAGES_DIR = os.path.join(POG_BASE_DIR, "images_sampled")

# 输出路径
OUTPUT_DIR = os.path.join(POG_BASE_DIR, "processed_dataset")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")

# 划分配置
TRAIN_RATIO = 0.8
VAL_SAMPLE_COUNT = 15  # 验证集固定样本数
MAX_TEST_SAMPLES = 2000  # 测试集最大样本数
TEST_RATIO = 0.15      # 剩余的作为测试集（用于划分用户）


def load_user_data() -> Dict[str, Dict]:
    """加载用户交互数据"""
    print("=" * 80)
    print("Loading user interaction data...")
    print("=" * 80)
    print(f"Filtering: users must have at least {MIN_SEQUENCE_LEN} items")
    
    if not os.path.exists(USER_DATA_FILE):
        raise FileNotFoundError(f"User data file not found at {USER_DATA_FILE}")
    
    user_data = {}
    filtered_count = 0
    
    print(f"  - Reading file: {USER_DATA_FILE}")
    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 格式: user_id,item_ids_separated_by_semicolon,outfit_id(忽略)
                parts = line.split(',')
                if len(parts) < 2:
                    continue
                
                user_id = parts[0].strip()
                item_ids_str = parts[1].strip()
                # parts[2] 是 outfit_id，不使用
                
                # 解析物品ID列表（分号分隔）
                item_ids = [item_id.strip() for item_id in item_ids_str.split(';') if item_id.strip()]
                
                # 过滤：至少需要 4 个 item（3个历史+1个target）
                if len(item_ids) < MIN_SEQUENCE_LEN:
                    filtered_count += 1
                    continue
                
                user_data[user_id] = {
                    'all_item_ids': item_ids  # 完整序列
                }
                
            except Exception as e:
                print(f"  - Warning: Error parsing line {line_num}: {e}, skipping...")
                continue
    
    print(f"  - Loaded {len(user_data):,} valid users")
    print(f"  - Filtered out {filtered_count:,} users (< {MIN_SEQUENCE_LEN} items)")
    
    # 统计信息
    total_items = sum(len(data['all_item_ids']) for data in user_data.values())
    avg_items = total_items / len(user_data) if user_data else 0
    max_items = max(len(data['all_item_ids']) for data in user_data.values()) if user_data else 0
    min_items = min(len(data['all_item_ids']) for data in user_data.values()) if user_data else 0
    
    print(f"  - Total items: {total_items:,}")
    print(f"  - Avg items per user: {avg_items:.1f}")
    print(f"  - Max items per user: {max_items}")
    print(f"  - Min items per user: {min_items}")
    
    # 预估样本数（滑动窗口：n个item生成 n-3 个样本）
    estimated_samples = sum(max(0, len(data['all_item_ids']) - HISTORY_LEN) for data in user_data.values())
    print(f"  - Estimated samples (sliding window): {estimated_samples:,}")
    
    return user_data


def load_captions() -> Dict[str, str]:
    """加载物品captions"""
    print("\n" + "=" * 80)
    print("Loading item captions...")
    print("=" * 80)
    
    if not os.path.exists(CAPTIONS_FILE):
        print(f"  - Warning: Captions file not found at {CAPTIONS_FILE}, returning empty dict")
        return {}
    
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"  - Loaded {len(captions):,} captions")
    return captions


def load_user_styles() -> Dict[str, str]:
    """加载用户风格关键词"""
    print("\n" + "=" * 80)
    print("Loading user styles...")
    print("=" * 80)
    
    if not os.path.exists(USER_STYLES_FILE):
        print(f"  - Warning: User styles file not found at {USER_STYLES_FILE}, returning empty dict")
        return {}
    
    with open(USER_STYLES_FILE, 'r', encoding='utf-8') as f:
        styles_data = json.load(f)
    
    # 转换为字典格式：user_id -> style_keywords
    user_styles = {}
    if isinstance(styles_data, list):
        for item in styles_data:
            if 'user' in item and 'style' in item:
                user_styles[item['user']] = item['style']
    elif isinstance(styles_data, dict):
        user_styles = styles_data
    
    print(f"  - Loaded styles for {len(user_styles):,} users")
    return user_styles


def find_image_path(item_id: str) -> str:
    """查找物品图片路径（尝试更多扩展名）"""
    if not os.path.exists(IMAGES_DIR):
        return None
    
    # 尝试更多图片扩展名（包括大小写变体和常见格式）
    extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.TIF',
        '.Jpg', '.Jpeg', '.Png', '.Gif', '.Bmp', '.Webp', '.Tiff', '.Tif'
    ]
    
    for ext in extensions:
        image_path = os.path.join(IMAGES_DIR, item_id + ext)
        if os.path.exists(image_path):
            return image_path
    
    return None


def create_dataset_samples(user_data: Dict[str, Dict], captions: Dict[str, str], 
                          user_styles: Dict[str, str]) -> List[Dict]:
    """创建数据集样本（固定窗口滑动：history_len=3, target=1）
    只保留所有图片（历史3个+目标1个）都存在的样本
    """
    print("\n" + "=" * 80)
    print("Creating dataset samples with sliding window...")
    print("=" * 80)
    print(f"  - Window size: history={HISTORY_LEN}, target=1")
    print(f"  - Filter: Only keeping samples with all 4 images available")
    
    samples = []
    users_with_samples = 0
    total_potential_samples = 0
    filtered_by_missing_images = 0
    
    for user_id, data in user_data.items():
        all_item_ids = data['all_item_ids']
        
        # 至少需要4个item
        if len(all_item_ids) < MIN_SEQUENCE_LEN:
            continue
        
        user_has_samples = False
        
        # 滑动窗口生成样本
        # 例如 [a,b,c,d,e,f] -> (a,b,c)->d, (b,c,d)->e, (c,d,e)->f
        for i in range(len(all_item_ids) - HISTORY_LEN):
            total_potential_samples += 1
            
            history_item_ids = all_item_ids[i:i+HISTORY_LEN]  # 固定取3个
            target_item_id = all_item_ids[i+HISTORY_LEN]      # 第4个作为target
            
            # === 验证所有图片是否存在 ===
            # 1. 检查历史3个图片
            history_image_paths = []
            all_history_images_exist = True
            for item_id in history_item_ids:
                img_path = find_image_path(item_id)
                if img_path is None:
                    all_history_images_exist = False
                    break
                history_image_paths.append(img_path)
            
            if not all_history_images_exist:
                filtered_by_missing_images += 1
                continue
            
            # 2. 检查目标图片
            target_image_path = find_image_path(target_item_id)
            if target_image_path is None:
                filtered_by_missing_images += 1
                continue
            
            # === 所有图片都存在，构建样本 ===
            history_items_info = []
            for idx, item_id in enumerate(history_item_ids):
                caption = captions.get(item_id, "")
                
                history_items_info.append({
                    'item_id': item_id,
                    'caption': caption,
                    'image_path': history_image_paths[idx]  # 使用已验证的路径
                })
            
            # 获取目标物品信息
            target_caption = captions.get(target_item_id, "")
            target_item_info = {
                'item_id': target_item_id,
                'caption': target_caption,
                'image_path': target_image_path  # 使用已验证的路径
            }
            
            # 获取用户风格
            user_style = user_styles.get(user_id, "")
            
            sample = {
                'user_id': user_id,
                'history_item_ids': history_item_ids,
                'history_items_info': history_items_info,
                'target_item_id': target_item_id,
                'target_item_info': target_item_info,
                'user_style': user_style,
                'num_interactions': HISTORY_LEN,
                'window_position': i,
                'total_sequence_length': len(all_item_ids)
            }
            
            samples.append(sample)
            user_has_samples = True
        
        if user_has_samples:
            users_with_samples += 1
    
    print(f"  - Total potential samples: {total_potential_samples:,}")
    print(f"  - Filtered by missing images: {filtered_by_missing_images:,} ({filtered_by_missing_images/total_potential_samples*100:.1f}%)")
    print(f"  - Created {len(samples):,} valid samples from {users_with_samples:,} users")
    if users_with_samples > 0:
        print(f"  - Average samples per user: {len(samples)/users_with_samples:.1f}")
    
    # 统计样本数分布
    samples_per_user = Counter(s['user_id'] for s in samples)
    sample_counts = Counter(samples_per_user.values())
    print(f"  - Sample distribution:")
    for count in sorted(sample_counts.keys())[:10]:
        print(f"      {count} sample(s): {sample_counts[count]:,} users")
    if len(sample_counts) > 10:
        print(f"      ... (showing first 10)")
    
    # 统计有caption和style的样本
    samples_with_history_captions = sum(1 for s in samples 
                                         if any(item.get('caption') for item in s['history_items_info']))
    samples_with_target_captions = sum(1 for s in samples if s['target_item_info'].get('caption'))
    samples_with_user_styles = sum(1 for s in samples if s.get('user_style'))
    
    print(f"\n  - Data coverage (all samples have images):")
    print(f"      History captions: {samples_with_history_captions:,}/{len(samples):,} ({samples_with_history_captions/len(samples)*100:.1f}%)")
    print(f"      Target captions: {samples_with_target_captions:,}/{len(samples):,} ({samples_with_target_captions/len(samples)*100:.1f}%)")
    print(f"      User styles: {samples_with_user_styles:,}/{len(samples):,} ({samples_with_user_styles/len(samples)*100:.1f}%)")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集为train/val/test（按用户划分，val固定15条，test最多2000条且均衡）"""
    print("\n" + "=" * 80)
    print("Splitting dataset by users...")
    print("=" * 80)
    print(f"  - Validation samples: {VAL_SAMPLE_COUNT} (fixed)")
    print(f"  - Test samples: max {MAX_TEST_SAMPLES} (balanced per user)")
    
    # 设置随机种子保证划分一致性
    random.seed(RANDOM_SEED)
    
    # 按用户划分，确保同一用户的所有样本在同一集合中
    user_to_samples = defaultdict(list)
    for sample in samples:
        user_to_samples[sample['user_id']].append(sample)
    
    user_ids = list(user_to_samples.keys())
    random.shuffle(user_ids)
    
    total_users = len(user_ids)
    
    # === 第一步：分出验证集（固定15条）===
    val_samples = []
    val_user_ids = []
    for uid in user_ids:
        if len(val_samples) >= VAL_SAMPLE_COUNT:
            break
        val_user_ids.append(uid)
        val_samples.extend(user_to_samples[uid])
    
    # 严格截断到15条
    if len(val_samples) > VAL_SAMPLE_COUNT:
        val_samples = val_samples[:VAL_SAMPLE_COUNT]
        val_user_ids_actual = list(set(s['user_id'] for s in val_samples))
    else:
        val_user_ids_actual = val_user_ids
    
    val_user_ids = set(val_user_ids_actual)
    
    # === 第二步：剩余用户划分 train 和 test ===
    remaining_users = [uid for uid in user_ids if uid not in val_user_ids]
    
    # 计算用于 test 的用户数量（约 15%）
    test_user_count = max(1, int(len(remaining_users) * TEST_RATIO / (TRAIN_RATIO + TEST_RATIO)))
    test_user_ids = set(remaining_users[-test_user_count:])  # 取后 15%
    train_user_ids = set(remaining_users[:-test_user_count])
    
    # === 第三步：从 test users 中均衡采样，最多 2000 条 ===
    # 收集所有 test user 的样本
    test_users_all_samples = defaultdict(list)
    total_test_samples_available = 0
    for uid in test_user_ids:
        test_users_all_samples[uid] = user_to_samples[uid]
        total_test_samples_available += len(user_to_samples[uid])
    
    print(f"\n  - Test users: {len(test_user_ids)}")
    print(f"  - Total test samples available: {total_test_samples_available:,}")
    
    test_samples = []
    
    if total_test_samples_available <= MAX_TEST_SAMPLES:
        # 如果总数不超过 2000，全部使用
        for uid in test_user_ids:
            test_samples.extend(test_users_all_samples[uid])
        print(f"  - Using all test samples: {len(test_samples):,}")
    else:
        # 需要均衡采样
        # 策略：每个用户按比例贡献，但设置上下限
        samples_per_user = MAX_TEST_SAMPLES / len(test_user_ids)
        
        # 第一轮：给每个用户分配基础配额
        for uid in test_user_ids:
            available = len(test_users_all_samples[uid])
            # 取 min(available, ceil(samples_per_user))
            quota = min(available, max(1, int(samples_per_user) + 1))
            sampled = random.sample(test_users_all_samples[uid], quota)
            test_samples.extend(sampled)
        
        # 如果超过 2000，随机截断
        if len(test_samples) > MAX_TEST_SAMPLES:
            random.shuffle(test_samples)
            test_samples = test_samples[:MAX_TEST_SAMPLES]
        
        print(f"  - Balanced sampling: {len(test_samples):,} samples")
        
        # 统计每个用户实际贡献的样本数
        test_user_contributions = Counter(s['user_id'] for s in test_samples)
        contributions = list(test_user_contributions.values())
        print(f"  - Samples per test user: min={min(contributions)}, max={max(contributions)}, "
              f"avg={sum(contributions)/len(contributions):.1f}")
    
    # === 第四步：收集 train samples ===
    train_samples = []
    for uid in train_user_ids:
        train_samples.extend(user_to_samples[uid])
    
    # === 统计信息 ===
    print(f"\n  - Total users: {total_users}")
    print(f"  - Train users: {len(train_user_ids)} ({len(train_user_ids)/total_users*100:.1f}%)")
    print(f"  - Val users: {len(val_user_ids)} ({len(val_user_ids)/total_users*100:.1f}%)")
    print(f"  - Test users: {len(test_user_ids)} ({len(test_user_ids)/total_users*100:.1f}%)")
    
    print(f"\n  - Total samples: {len(samples):,}")
    print(f"  - Train samples: {len(train_samples):,} ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"  - Val samples: {len(val_samples):,} (fixed at {VAL_SAMPLE_COUNT})")
    print(f"  - Test samples: {len(test_samples):,} ({len(test_samples)/len(samples)*100:.1f}%)")
    
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
    
    print(f"\n[2] Fixed Window Configuration:")
    print(f"  - History length: {HISTORY_LEN} (fixed)")
    print(f"  - Target: 1 item (fixed)")
    print(f"  - Min sequence length: {MIN_SEQUENCE_LEN} items")
    print(f"  - Image filter: All 4 images must exist")
    
    print(f"\n[3] Sample Data Structure (example):")
    if all_samples:
        sample = all_samples[0]
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - user_id: {sample['user_id']}")
        print(f"  - num_interactions: {sample['num_interactions']} (always {HISTORY_LEN})")
        print(f"  - window_position: {sample.get('window_position', 'N/A')}")
        print(f"  - total_sequence_length: {sample.get('total_sequence_length', 'N/A')}")
        print(f"  - has_target_item: {sample.get('target_item_id') is not None}")
        print(f"  - has_user_style: {bool(sample.get('user_style'))}")
        print(f"  - History items (always 3, all have images):")
        for i, item in enumerate(sample['history_items_info']):
            print(f"      [{i+1}] item_id: {item['item_id']}, "
                  f"has_caption: {bool(item.get('caption'))}, "
                  f"image: {item.get('image_path', 'N/A')[-30:]}")  # 显示路径末尾
        print(f"  - Target item (has image):")
        target = sample['target_item_info']
        print(f"      item_id: {target['item_id']}, "
              f"has_caption: {bool(target.get('caption'))}, "
              f"image: {target.get('image_path', 'N/A')[-30:]}")
    
    print(f"\n[4] User Distribution:")
    user_sample_counts = {}
    for s in all_samples:
        uid = s['user_id']
        user_sample_counts[uid] = user_sample_counts.get(uid, 0) + 1
    
    if user_sample_counts:
        avg_samples_per_user = sum(user_sample_counts.values()) / len(user_sample_counts)
        print(f"  - Total unique users: {len(user_sample_counts):,}")
        print(f"  - Average samples per user: {avg_samples_per_user:.1f}")
        print(f"  - Users with 1 sample: {sum(1 for c in user_sample_counts.values() if c == 1):,}")
        print(f"  - Users with 2-5 samples: {sum(1 for c in user_sample_counts.values() if 2 <= c <= 5):,}")
        print(f"  - Users with 6-10 samples: {sum(1 for c in user_sample_counts.values() if 6 <= c <= 10):,}")
        print(f"  - Users with 10+ samples: {sum(1 for c in user_sample_counts.values() if c > 10):,}")


def main():
    """主函数"""
    print("=" * 80)
    print("POG Dataset Processing (Fixed Sliding Window)")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Window: history={HISTORY_LEN}, target=1")
    print(f"Image filter: All 4 images must exist")
    print(f"Split config: Train={TRAIN_RATIO}, Val={VAL_SAMPLE_COUNT} samples, Test=max {MAX_TEST_SAMPLES} samples")
    
    # 1. 加载用户交互数据
    user_data = load_user_data()
    
    # 2. 加载captions
    captions = load_captions()
    
    # 3. 加载用户风格
    user_styles = load_user_styles()
    
    # 4. 创建数据集样本（固定窗口滑动，过滤缺失图片）
    samples = create_dataset_samples(user_data, captions, user_styles)
    
    # 5. 划分数据集
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    # 6. 保存数据集
    save_datasets(train_samples, val_samples, test_samples)
    
    # 7. 打印统计信息
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