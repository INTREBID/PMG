"""
Unified Dataset classes for PMG
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Optional


class PMGDataset(Dataset):
    """
    Unified dataset for PMG training across FLICKR, POG, and SER datasets
    
    Args:
        data: List of training samples
        tokenizer: LLaMA tokenizer
        sd_pipeline: Stable Diffusion pipeline for text encoding
        dataset_name: Name of dataset ("FLICKR", "POG", or "SER")
        max_len: Maximum sequence length
        num_image_prompt: Number of image prompt tokens
        image_size: Image size for resizing
        mode: 'train' or 'val'
        repeats: Number of times to repeat the dataset
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        sd_pipeline,
        dataset_name: str = "FLICKR",
        max_len: int = 600,
        num_image_prompt: int = 2,
        image_size: int = 512,
        mode: str = 'train',
        repeats: int = 1
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.sd_pipeline = sd_pipeline
        self.dataset_name = dataset_name.upper()
        self.max_len = max_len
        self.num_image_prompt = num_image_prompt
        self.image_size = image_size
        self.mode = mode
        self._length = len(data) * repeats
        
        # Precompute embeddings for all unique items
        self._precompute_embeddings()
    
    def _precompute_embeddings(self):
        """Precompute text embeddings for all unique items"""
        self.item_emb_dict = {}
        self.item_ids_dict = {}
        
        # Collect all unique items
        unique_items = {}
        for sample in self.data:
            # History items
            for item_info in sample.get('history_items_info', []):
                item_id = item_info.get('item_id')
                caption = item_info.get('caption')
                if item_id and caption and item_id not in unique_items:
                    unique_items[item_id] = caption
            
            # Target item
            target_info = sample.get('target_item_info', {})
            item_id = target_info.get('item_id')
            caption = target_info.get('caption')
            if item_id and caption and item_id not in unique_items:
                unique_items[item_id] = caption
        
        print(f"[{self.mode}] Computing embeddings for {len(unique_items)} unique items...")
        items_list = list(unique_items.items())
        
        # Batch encoding for efficiency
        batch_size = 512
        for i in tqdm(range(0, len(items_list), batch_size), desc=f"[{self.mode}] Encoding"):
            batch_items = items_list[i:i+batch_size]
            batch_ids = [item_id for item_id, _ in batch_items]
            batch_captions = [caption for _, caption in batch_items]
            
            # Encode with SD text encoder
            tokens_list = []
            for caption in batch_captions:
                tokens = self.sd_pipeline.textEncode(
                    caption, num_tokens=75, return_tokens=True
                ).detach()[0]
                tokens_list.append(tokens)
            
            embs = self.sd_pipeline.textEncode(tokens=torch.stack(tokens_list, dim=0))
            
            # Store embeddings
            for j, item_id in enumerate(batch_ids):
                self.item_ids_dict[item_id] = tokens_list[j].cpu()
                self.item_emb_dict[item_id] = embs[j].detach().cpu()
    
    def _resize_rgb(self, img_path: str) -> np.ndarray:
        """Load and resize image from path"""
        if not img_path or not os.path.exists(img_path):
            # Return random image as fallback
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        try:
            im = Image.open(img_path).convert('RGB')
            im = im.resize((self.image_size, self.image_size), Image.BICUBIC)
            arr = np.array(im, dtype=np.uint8)
            return np.ascontiguousarray(arr)
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            return np.random.randint(0, 255, (self.image_size, self.image_size, 3), dtype=np.uint8)
    
    def _build_prompt(self, history_items: List[Dict]) -> str:
        """Build LLaMA prompt from history items"""
        history_captions = [
            f"{k+1}. {item['caption']}"
            for k, item in enumerate(history_items)
            if item.get('caption')
        ]
        history_text = " ".join(history_captions)
        
        prompt = (
            "### Human: A person rated the following images highly: \"<Images/>\". "
            "Describe their visual taste. ###Assistant: "
        )
        prompt = prompt.replace('<Images/>', history_text)
        
        return prompt
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        """Get a training sample"""
        idx = idx % len(self.data)
        sample = self.data[idx]
        
        # Extract user/worker ID based on dataset
        user_id = sample.get('user_id') or sample.get('worker_id', '')
        history_items = sample.get('history_items_info', [])
        target_item = sample.get('target_item_info', {})
        user_style = sample.get('user_style') or sample.get('worker_style', '')
        
        # Build prompt for LLaMA
        prompt_text = self._build_prompt(history_items)
        product_token = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        
        # Prepare example
        example = {}
        example['token_len'] = len(product_token)
        
        # Pad tokens
        assert self.max_len >= len(product_token) + self.num_image_prompt, \
            f"Sequence too long: {len(product_token)} + {self.num_image_prompt} > {self.max_len}"
        
        product_token += [self.tokenizer.pad_token_id] * (self.max_len - len(product_token))
        example['input_ids'] = torch.tensor(product_token)
        
        # Target item (positive sample)
        target_id = target_item.get('item_id', '')
        example['keywords_ids'] = self.item_ids_dict.get(
            target_id, torch.zeros(77, dtype=torch.long)
        )
        example['keywords_emb'] = self.item_emb_dict.get(
            target_id, torch.zeros(77, 768)
        )
        example['pixel_values'] = self._resize_rgb(target_item.get('image_path', ''))
        
        # Negative sample (random other item)
        nega_id = np.random.choice(list(self.item_emb_dict.keys()))
        example['nega_keywords_ids'] = self.item_ids_dict[nega_id]
        example['nega_keywords_emb'] = self.item_emb_dict[nega_id]
        
        # Find negative sample image path
        nega_img_path = None
        for s in self.data:
            if s.get('target_item_info', {}).get('item_id') == nega_id:
                nega_img_path = s['target_item_info'].get('image_path')
                break
        
        if not nega_img_path:
            # Search in history
            for s in self.data:
                for h in s.get('history_items_info', []):
                    if h.get('item_id') == nega_id:
                        nega_img_path = h.get('image_path')
                        break
                if nega_img_path:
                    break
        
        example['nega_pixel_values'] = self._resize_rgb(nega_img_path or '')
        
        return example


class PMGInferenceDataset(Dataset):
    """
    Dataset for PMG inference (generation)
    
    Args:
        data: List of test samples
        tokenizer: LLaMA tokenizer
        max_len: Maximum sequence length
        dataset_name: Name of dataset
    """
    
    def __init__(
        self,
        data: List[Dict],
        tokenizer,
        max_len: int = 600,
        dataset_name: str = "FLICKR"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.dataset_name = dataset_name.upper()
    
    def _build_prompt(self, history_items: List[Dict]) -> str:
        """Build LLaMA prompt from history items"""
        history_captions = [
            f"{k+1}. {item['caption']}"
            for k, item in enumerate(history_items)
            if item.get('caption')
        ]
        history_text = " ".join(history_captions)
        
        prompt = (
            "### Human: A person rated the following images highly: \"<Images/>\". "
            "Describe their visual taste. ###Assistant: "
        )
        prompt = prompt.replace('<Images/>', history_text)
        
        return prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get an inference sample"""
        sample = self.data[idx]
        
        # Extract basic info
        user_id = sample.get('user_id') or sample.get('worker_id', '')
        history_items = sample.get('history_items_info', [])
        target_item = sample.get('target_item_info', {})
        
        # Build prompt
        prompt_text = self._build_prompt(history_items)
        product_token = self.tokenizer(prompt_text, return_tensors="pt").input_ids[0].tolist()
        
        # Prepare example
        example = {}
        example['token_len'] = len(product_token)
        example['sample_idx'] = idx
        example['user_id'] = user_id
        example['target_item_id'] = target_item.get('item_id', '')
        example['target_caption'] = target_item.get('caption', '')
        
        # Pad tokens
        product_token += [self.tokenizer.pad_token_id] * (self.max_len - len(product_token))
        example['input_ids'] = torch.tensor(product_token)
        
        return example

