#!/usr/bin/env python3
import os
import json
import csv
import random
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

# 固定随机种子
RANDOM_SEED = 42

# 固定窗口大小
HISTORY_LEN = 3  # 每个样本的历史长度
MIN_SEQUENCE_LEN = HISTORY_LEN + 1  # 最少需要的item数量（3个历史+1个target=4）

# 评分过滤阈值
MIN_WORKER_SCORE = 2  # 只保留评分 > 2 的交互

# 数据集路径
FLICKR_BASE_DIR = "/data-nfs/gpu1-1/ud202581869/Personalized_Generation/FLICKR"
CSV_FILE = os.path.join(FLICKR_BASE_DIR, "FLICKR-AES_image_labeled_by_each_worker.csv")
CAPTIONS_FILE = os.path.join(FLICKR_BASE_DIR, "FLICKR_captions.json")
WORKER_STYLES_FILE = os.path.join(FLICKR_BASE_DIR, "FLICKR_styles.json")
IMAGE_SCORE_FILE = os.path.join(FLICKR_BASE_DIR, "FLICKR-AES_image_score.txt")
IMAGE_DIR = os.path.join(FLICKR_BASE_DIR, "40K")

# 输出路径
OUTPUT_DIR = os.path.join(FLICKR_BASE_DIR, "processed_dataset")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "train.json")
OUTPUT_VAL = os.path.join(OUTPUT_DIR, "val.json")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "test.json")

# 划分配置
TRAIN_RATIO = 0.8
VAL_SAMPLE_COUNT = 15  # 验证集固定样本数
MAX_TEST_SAMPLES = 2000  # 测试集最大样本数
TEST_RATIO = 0.15      # 剩余的作为测试集（用于划分 worker）


def load_flickr_interactions() -> Dict[str, List[Dict]]:
    """从 CSV 加载 worker 的交互数据（只保留评分 > 2 的）"""
    print("=" * 80)
    print("Loading FLICKR interaction data from CSV...")
    print("=" * 80)
    print(f"  - Filtering: Only keeping interactions with score > {MIN_WORKER_SCORE}")
    
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"CSV file not found: {CSV_FILE}")
    
    worker_interactions = defaultdict(list)
    total_interactions = 0
    filtered_by_score = 0
    
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        
        for row in reader:
            total_interactions += 1
            worker_id = row['worker'].strip()
            image_id = row[' imagePair'].strip() if ' imagePair' in row else row['imagePair'].strip()
            score = int(row['score'])
            
            # 只保留评分 > 2 的交互
            if score <= MIN_WORKER_SCORE:
                filtered_by_score += 1
                continue
            
            # 去掉 .jpg 扩展名
            image_id_clean = os.path.splitext(image_id)[0]
            
            worker_interactions[worker_id].append({
                'item_id': image_id_clean,
                'score': score
            })
    
    print(f"  - Total interactions in CSV: {total_interactions:,}")
    print(f"  - Filtered by score (≤ {MIN_WORKER_SCORE}): {filtered_by_score:,} ({filtered_by_score/total_interactions*100:.1f}%)")
    kept_interactions = sum(len(items) for items in worker_interactions.values())
    print(f"  - Kept interactions (> {MIN_WORKER_SCORE}): {kept_interactions:,} ({kept_interactions/total_interactions*100:.1f}%)")
    print(f"  - Total workers with high-score interactions: {len(worker_interactions):,}")
    
    # 过滤：至少4个交互
    filtered_workers = {}
    filtered_count = 0
    for worker_id, items in worker_interactions.items():
        if len(items) >= MIN_SEQUENCE_LEN:
            # 按时间顺序（CSV 中的顺序）保留
            filtered_workers[worker_id] = items
        else:
            filtered_count += 1
    
    print(f"  - Workers with >= {MIN_SEQUENCE_LEN} high-score interactions: {len(filtered_workers):,}")
    print(f"  - Filtered out: {filtered_count:,} workers (insufficient interactions)")
    
    # 统计
    total_items = sum(len(items) for items in filtered_workers.values())
    avg_items = total_items / len(filtered_workers) if filtered_workers else 0
    print(f"  - Total interactions (after all filters): {total_items:,}")
    print(f"  - Avg interactions per worker: {avg_items:.1f}")
    
    estimated_samples = sum(max(0, len(items) - HISTORY_LEN) for items in filtered_workers.values())
    print(f"  - Estimated samples (sliding window): {estimated_samples:,}")
    
    # 统计评分分布
    all_scores = [item['score'] for items in filtered_workers.values() for item in items]
    score_dist = Counter(all_scores)
    print(f"\n  - Score distribution (kept interactions):")
    for score in sorted(score_dist.keys()):
        print(f"      Score {score}: {score_dist[score]:,} ({score_dist[score]/len(all_scores)*100:.1f}%)")
    
    return filtered_workers


def load_captions() -> Dict[str, str]:
    """加载图片 captions"""
    print("\n" + "=" * 80)
    print("Loading captions...")
    print("=" * 80)
    
    if not os.path.exists(CAPTIONS_FILE):
        print(f"  - Warning: {CAPTIONS_FILE} not found, returning empty dict")
        return {}
    
    with open(CAPTIONS_FILE, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    print(f"  - Loaded {len(captions):,} captions")
    return captions


def load_worker_styles() -> Dict[str, str]:
    """加载 worker 风格关键词"""
    print("\n" + "=" * 80)
    print("Loading worker styles...")
    print("=" * 80)
    
    if not os.path.exists(WORKER_STYLES_FILE):
        print(f"  - Warning: {WORKER_STYLES_FILE} not found, returning empty dict")
        return {}
    
    with open(WORKER_STYLES_FILE, 'r', encoding='utf-8') as f:
        styles_data = json.load(f)
    
    # 转换为字典格式：worker_id -> style_keywords
    worker_styles = {}
    if isinstance(styles_data, list):
        for item in styles_data:
            if 'worker' in item and 'style' in item:
                worker_styles[item['worker']] = item['style']
    elif isinstance(styles_data, dict):
        worker_styles = styles_data
    
    print(f"  - Loaded styles for {len(worker_styles):,} workers")
    return worker_styles


def load_image_scores() -> Dict[str, float]:
    """加载图片美学评分"""
    print("\n" + "=" * 80)
    print("Loading image aesthetic scores...")
    print("=" * 80)
    
    if not os.path.exists(IMAGE_SCORE_FILE):
        print(f"  - Warning: {IMAGE_SCORE_FILE} not found, returning empty dict")
        return {}
    
    image_scores = {}
    with open(IMAGE_SCORE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                image_id = os.path.splitext(parts[0])[0]  # 去掉扩展名
                score = float(parts[1])
                image_scores[image_id] = score
    
    print(f"  - Loaded aesthetic scores for {len(image_scores):,} images")
    return image_scores


def find_image_path(item_id: str) -> str:
    """查找图片路径（尝试更多扩展名）"""
    if not os.path.exists(IMAGE_DIR):
        return None
    
    # 尝试更多图片扩展名（包括大小写变体和常见格式）
    extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif',
        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.WEBP', '.TIFF', '.TIF',
        '.Jpg', '.Jpeg', '.Png', '.Gif', '.Bmp', '.Webp', '.Tiff', '.Tif'
    ]
    
    for ext in extensions:
        image_path = os.path.join(IMAGE_DIR, item_id + ext)
        if os.path.exists(image_path):
            return image_path
    
    return None


def create_dataset_samples(worker_data: Dict[str, List[Dict]], 
                          captions: Dict[str, str],
                          worker_styles: Dict[str, str],
                          image_scores: Dict[str, float]) -> List[Dict]:
    """创建数据集样本（固定窗口滑动：history_len=3, target=1）
    只保留所有图片（历史3个+目标1个）都存在的样本
    """
    print("\n" + "=" * 80)
    print("Creating dataset samples with sliding window...")
    print("=" * 80)
    print(f"  - Window size: history={HISTORY_LEN}, target=1")
    print(f"  - Filter: Only keeping samples with all 4 images available")
    
    samples = []
    workers_with_samples = 0
    total_potential_samples = 0
    filtered_by_missing_images = 0
    
    for worker_id, interactions in worker_data.items():
        # 至少需要4个交互
        if len(interactions) < MIN_SEQUENCE_LEN:
            continue
        
        worker_has_samples = False
        
        # 滑动窗口生成样本
        # 例如 [a,b,c,d,e,f] -> (a,b,c)->d, (b,c,d)->e, (c,d,e)->f
        for i in range(len(interactions) - HISTORY_LEN):
            total_potential_samples += 1
            
            history_items = interactions[i:i+HISTORY_LEN]  # 固定取3个
            target_item = interactions[i+HISTORY_LEN]      # 第4个作为target
            
            # === 验证所有图片是否存在 ===
            # 1. 检查历史3个图片
            history_image_paths = []
            all_history_images_exist = True
            for item in history_items:
                img_path = find_image_path(item['item_id'])
                if img_path is None:
                    all_history_images_exist = False
                    break
                history_image_paths.append(img_path)
            
            if not all_history_images_exist:
                filtered_by_missing_images += 1
                continue
            
            # 2. 检查目标图片
            target_image_path = find_image_path(target_item['item_id'])
            if target_image_path is None:
                filtered_by_missing_images += 1
                continue
            
            # === 所有图片都存在，构建样本 ===
            history_items_info = []
            history_item_ids = []
            for idx, item in enumerate(history_items):
                item_id = item['item_id']
                history_item_ids.append(item_id)
                caption = captions.get(item_id, "")
                aesthetic_score = image_scores.get(item_id, 0.0)
                
                history_items_info.append({
                    'item_id': item_id,
                    'caption': caption,
                    'image_path': history_image_paths[idx],  # 使用已验证的路径
                    'score': item.get('score', 0),
                    'aesthetic_score': aesthetic_score
                })
            
            # 获取目标物品信息
            target_item_id = target_item['item_id']
            target_caption = captions.get(target_item_id, "")
            target_aesthetic_score = image_scores.get(target_item_id, 0.0)
            target_item_info = {
                'item_id': target_item_id,
                'caption': target_caption,
                'image_path': target_image_path,  # 使用已验证的路径
                'score': target_item.get('score', 0),
                'aesthetic_score': target_aesthetic_score
            }
            
            # 获取 worker 风格
            worker_style = worker_styles.get(worker_id, "")
            
            sample = {
                'worker_id': worker_id,
                'history_item_ids': history_item_ids,
                'history_items_info': history_items_info,
                'target_item_id': target_item_id,
                'target_item_info': target_item_info,
                'worker_style': worker_style,
                'num_interactions': HISTORY_LEN,
                'window_position': i,
                'total_sequence_length': len(interactions)
            }
            
            samples.append(sample)
            worker_has_samples = True
        
        if worker_has_samples:
            workers_with_samples += 1
    
    print(f"  - Total potential samples: {total_potential_samples:,}")
    print(f"  - Filtered by missing images: {filtered_by_missing_images:,} ({filtered_by_missing_images/total_potential_samples*100:.1f}%)")
    print(f"  - Created {len(samples):,} valid samples from {workers_with_samples:,} workers")
    if workers_with_samples > 0:
        print(f"  - Average samples per worker: {len(samples)/workers_with_samples:.1f}")
    
    # 统计样本数分布
    samples_per_worker = Counter(s['worker_id'] for s in samples)
    sample_counts = Counter(samples_per_worker.values())
    print(f"  - Sample distribution:")
    for count in sorted(sample_counts.keys())[:10]:
        print(f"      {count} sample(s): {sample_counts[count]:,} workers")
    if len(sample_counts) > 10:
        print(f"      ... (showing first 10)")
    
    # 统计有caption和style的样本
    samples_with_history_captions = sum(1 for s in samples 
                                         if any(item.get('caption') for item in s['history_items_info']))
    samples_with_target_captions = sum(1 for s in samples if s['target_item_info'].get('caption'))
    samples_with_worker_styles = sum(1 for s in samples if s.get('worker_style'))
    
    print(f"\n  - Data coverage (all samples have images):")
    print(f"      History captions: {samples_with_history_captions:,}/{len(samples):,} ({samples_with_history_captions/len(samples)*100:.1f}%)")
    print(f"      Target captions: {samples_with_target_captions:,}/{len(samples):,} ({samples_with_target_captions/len(samples)*100:.1f}%)")
    print(f"      Worker styles: {samples_with_worker_styles:,}/{len(samples):,} ({samples_with_worker_styles/len(samples)*100:.1f}%)")
    
    return samples


def split_dataset(samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """划分数据集为train/val/test（按 worker 划分，val固定15条，test最多2000条且均衡）"""
    print("\n" + "=" * 80)
    print("Splitting dataset by workers...")
    print("=" * 80)
    print(f"  - Validation samples: {VAL_SAMPLE_COUNT} (fixed)")
    print(f"  - Test samples: max {MAX_TEST_SAMPLES} (balanced per worker)")
    
    # 设置随机种子保证划分一致性
    random.seed(RANDOM_SEED)
    
    # 按 worker 划分，确保同一 worker 的所有样本在同一集合中
    worker_to_samples = defaultdict(list)
    for sample in samples:
        worker_to_samples[sample['worker_id']].append(sample)
    
    worker_ids = list(worker_to_samples.keys())
    random.shuffle(worker_ids)
    
    total_workers = len(worker_ids)
    
    # === 第一步：分出验证集（固定15条）===
    val_samples = []
    val_worker_ids = []
    for uid in worker_ids:
        if len(val_samples) >= VAL_SAMPLE_COUNT:
            break
        val_worker_ids.append(uid)
        val_samples.extend(worker_to_samples[uid])
    
    # 严格截断到15条
    if len(val_samples) > VAL_SAMPLE_COUNT:
        val_samples = val_samples[:VAL_SAMPLE_COUNT]
        val_worker_ids_actual = list(set(s['worker_id'] for s in val_samples))
    else:
        val_worker_ids_actual = val_worker_ids
    
    val_worker_ids = set(val_worker_ids_actual)
    
    # === 第二步：剩余 worker 划分 train 和 test ===
    remaining_workers = [uid for uid in worker_ids if uid not in val_worker_ids]
    
    # 计算用于 test 的 worker 数量（约 15%）
    test_worker_count = max(1, int(len(remaining_workers) * TEST_RATIO / (TRAIN_RATIO + TEST_RATIO)))
    test_worker_ids = set(remaining_workers[-test_worker_count:])  # 取后 15%
    train_worker_ids = set(remaining_workers[:-test_worker_count])
    
    # === 第三步：从 test workers 中均衡采样，最多 2000 条 ===
    # 收集所有 test worker 的样本
    test_workers_all_samples = defaultdict(list)
    total_test_samples_available = 0
    for uid in test_worker_ids:
        test_workers_all_samples[uid] = worker_to_samples[uid]
        total_test_samples_available += len(worker_to_samples[uid])
    
    print(f"\n  - Test workers: {len(test_worker_ids)}")
    print(f"  - Total test samples available: {total_test_samples_available:,}")
    
    test_samples = []
    
    if total_test_samples_available <= MAX_TEST_SAMPLES:
        # 如果总数不超过 2000，全部使用
        for uid in test_worker_ids:
            test_samples.extend(test_workers_all_samples[uid])
        print(f"  - Using all test samples: {len(test_samples):,}")
    else:
        # 需要均衡采样
        # 策略：每个 worker 按比例贡献，但设置上下限
        samples_per_worker = MAX_TEST_SAMPLES / len(test_worker_ids)
        
        # 第一轮：给每个 worker 分配基础配额
        for uid in test_worker_ids:
            available = len(test_workers_all_samples[uid])
            # 取 min(available, ceil(samples_per_worker))
            quota = min(available, max(1, int(samples_per_worker) + 1))
            sampled = random.sample(test_workers_all_samples[uid], quota)
            test_samples.extend(sampled)
        
        # 如果超过 2000，随机截断
        if len(test_samples) > MAX_TEST_SAMPLES:
            random.shuffle(test_samples)
            test_samples = test_samples[:MAX_TEST_SAMPLES]
        
        print(f"  - Balanced sampling: {len(test_samples):,} samples")
        
        # 统计每个 worker 实际贡献的样本数
        test_worker_contributions = Counter(s['worker_id'] for s in test_samples)
        contributions = list(test_worker_contributions.values())
        print(f"  - Samples per test worker: min={min(contributions)}, max={max(contributions)}, "
              f"avg={sum(contributions)/len(contributions):.1f}")
    
    # === 第四步：收集 train samples ===
    train_samples = []
    for uid in train_worker_ids:
        train_samples.extend(worker_to_samples[uid])
    
    # === 统计信息 ===
    print(f"\n  - Total workers: {total_workers}")
    print(f"  - Train workers: {len(train_worker_ids)} ({len(train_worker_ids)/total_workers*100:.1f}%)")
    print(f"  - Val workers: {len(val_worker_ids)} ({len(val_worker_ids)/total_workers*100:.1f}%)")
    print(f"  - Test workers: {len(test_worker_ids)} ({len(test_worker_ids)/total_workers*100:.1f}%)")
    
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
    print(f"  - Min sequence length: {MIN_SEQUENCE_LEN} interactions")
    print(f"  - Score filter: Only score > {MIN_WORKER_SCORE}")
    print(f"  - Image filter: All 4 images must exist")
    
    print(f"\n[3] Sample Data Structure (example):")
    if all_samples:
        sample = all_samples[0]
        print(f"  - Keys: {list(sample.keys())}")
        print(f"  - worker_id: {sample['worker_id']}")
        print(f"  - num_interactions: {sample['num_interactions']} (always {HISTORY_LEN})")
        print(f"  - window_position: {sample.get('window_position', 'N/A')}")
        print(f"  - total_sequence_length: {sample.get('total_sequence_length', 'N/A')}")
        print(f"  - has_worker_style: {bool(sample.get('worker_style'))}")
        print(f"  - History items (always 3, all have images):")
        for i, item in enumerate(sample['history_items_info']):
            print(f"      [{i+1}] item_id: {item['item_id']}, score: {item.get('score', 'N/A')}, "
                  f"aesthetic: {item.get('aesthetic_score', 'N/A'):.3f}, "
                  f"has_caption: {bool(item.get('caption'))}, "
                  f"image: {item.get('image_path', 'N/A')[-30:]}")  # 显示路径末尾
        print(f"  - Target item (has image):")
        target = sample['target_item_info']
        print(f"      item_id: {target['item_id']}, score: {target.get('score', 'N/A')}, "
              f"aesthetic: {target.get('aesthetic_score', 'N/A'):.3f}, "
              f"has_caption: {bool(target.get('caption'))}, "
              f"image: {target.get('image_path', 'N/A')[-30:]}")
    
    print(f"\n[4] Worker Distribution:")
    worker_sample_counts = {}
    for s in all_samples:
        wid = s['worker_id']
        worker_sample_counts[wid] = worker_sample_counts.get(wid, 0) + 1
    
    if worker_sample_counts:
        avg_samples_per_worker = sum(worker_sample_counts.values()) / len(worker_sample_counts)
        print(f"  - Total unique workers: {len(worker_sample_counts):,}")
        print(f"  - Average samples per worker: {avg_samples_per_worker:.1f}")
        print(f"  - Workers with 1 sample: {sum(1 for c in worker_sample_counts.values() if c == 1):,}")
        print(f"  - Workers with 2-5 samples: {sum(1 for c in worker_sample_counts.values() if 2 <= c <= 5):,}")
        print(f"  - Workers with 6-10 samples: {sum(1 for c in worker_sample_counts.values() if 6 <= c <= 10):,}")
        print(f"  - Workers with 10+ samples: {sum(1 for c in worker_sample_counts.values() if c > 10):,}")


def main():
    """主函数"""
    print("=" * 80)
    print("FLICKR Dataset Processing (Fixed Sliding Window)")
    print("=" * 80)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Window: history={HISTORY_LEN}, target=1")
    print(f"Score filter: > {MIN_WORKER_SCORE}")
    print(f"Image filter: All 4 images must exist")
    print(f"Split config: Train={TRAIN_RATIO}, Val={VAL_SAMPLE_COUNT} samples, Test=max {MAX_TEST_SAMPLES} samples")
    
    # 1. 加载 worker 交互数据（从 CSV，过滤评分 > 2）
    worker_data = load_flickr_interactions()
    
    # 2. 加载captions
    captions = load_captions()
    
    # 3. 加载 worker 风格
    worker_styles = load_worker_styles()
    
    # 4. 加载图片美学评分
    image_scores = load_image_scores()
    
    # 5. 创建数据集样本（固定窗口滑动，过滤缺失图片）
    samples = create_dataset_samples(worker_data, captions, worker_styles, image_scores)
    
    # 6. 划分数据集
    train_samples, val_samples, test_samples = split_dataset(samples)
    
    # 7. 保存数据集
    save_datasets(train_samples, val_samples, test_samples)
    
    # 8. 打印统计信息
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