"""
Prompt generation utilities for PMG
"""
from typing import List, Dict, Optional


def generate_user_preference_prompt(
    history_items: List[Dict],
    style_keywords: Optional[str] = None,
    dataset: str = "FLICKR"
) -> str:
    """
    Generate a prompt describing user preferences based on history
    
    Args:
        history_items: List of historical items with captions
        style_keywords: User style keywords (if available)
        dataset: Dataset name for format-specific handling
        
    Returns:
        Formatted prompt string
    """
    if dataset == "FLICKR":
        prompt = "User preference based on aesthetic history:\n"
    elif dataset == "POG":
        prompt = "User fashion preference based on purchase history:\n"
    elif dataset == "SER":
        prompt = "User sticker preference based on interaction history:\n"
    else:
        prompt = "User preference based on history:\n"
    
    # Add history captions
    for i, item in enumerate(history_items[:5]):  # Limit to 5 items
        caption = item.get('caption', '')
        if caption:
            prompt += f"  {i+1}. {caption}\n"
    
    # Add style keywords if available
    if style_keywords:
        prompt += f"\nUser style: {style_keywords}\n"
    
    return prompt


def generate_target_prompt(
    target_caption: str,
    user_preferences: Optional[str] = None,
    dataset: str = "FLICKR"
) -> str:
    """
    Generate a prompt for target image generation
    
    Args:
        target_caption: Caption for the target item
        user_preferences: User preference text
        dataset: Dataset name
        
    Returns:
        Formatted prompt string
    """
    if user_preferences:
        prompt = f"Generate an image matching this description and user preference:\n"
        prompt += f"Description: {target_caption}\n"
        prompt += f"User preference: {user_preferences}"
    else:
        prompt = f"Generate an image: {target_caption}"
    
    return prompt


def format_llama_prompt(
    history_captions: List[str],
    target_caption: str,
    style_keywords: Optional[str] = None
) -> str:
    """
    Format prompt for LLaMA to extract user preferences
    
    Args:
        history_captions: List of historical item captions
        target_caption: Target item caption
        style_keywords: User style keywords
        
    Returns:
        Formatted LLaMA prompt
    """
    prompt = "[INST] Based on the following user interaction history, "
    prompt += "extract key visual and stylistic preferences, "
    prompt += "then generate a description for the target item.\n\n"
    
    prompt += "User History:\n"
    for i, caption in enumerate(history_captions):
        prompt += f"{i+1}. {caption}\n"
    
    if style_keywords:
        prompt += f"\nUser Style Keywords: {style_keywords}\n"
    
    prompt += f"\nTarget Item: {target_caption}\n"
    prompt += "\nPlease describe what kind of image would match the user's preference: [/INST]"
    
    return prompt


def extract_keywords_from_captions(captions: List[str], max_keywords: int = 10) -> str:
    """
    Extract keywords from a list of captions (simple version)
    
    Args:
        captions: List of caption strings
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        Comma-separated keywords string
    """
    # Simple keyword extraction: collect unique adjectives and nouns
    # This is a placeholder - could be replaced with more sophisticated NLP
    from collections import Counter
    import re
    
    # Common stop words to filter out
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'
    }
    
    # Extract words from all captions
    all_words = []
    for caption in captions:
        words = re.findall(r'\b[a-z]+\b', caption.lower())
        all_words.extend([w for w in words if w not in stop_words and len(w) > 3])
    
    # Count frequencies
    word_counts = Counter(all_words)
    
    # Get top keywords
    top_keywords = [word for word, count in word_counts.most_common(max_keywords)]
    
    return ", ".join(top_keywords)


def build_negative_prompt(dataset: str = "FLICKR") -> str:
    """
    Build negative prompt for Stable Diffusion
    
    Args:
        dataset: Dataset name
        
    Returns:
        Negative prompt string
    """
    common_negatives = [
        "blurry", "low quality", "distorted", "deformed", "ugly",
        "bad anatomy", "bad proportions", "watermark", "text"
    ]
    
    if dataset == "FLICKR":
        # For aesthetic photos, avoid artistic artifacts
        negatives = common_negatives + ["drawing", "painting", "cartoon", "illustration"]
    elif dataset == "POG":
        # For fashion, avoid non-clothing items
        negatives = common_negatives + ["face", "portrait", "landscape"]
    elif dataset == "SER":
        # For stickers, avoid photorealistic elements
        negatives = common_negatives + ["photorealistic", "3d", "realistic"]
    else:
        negatives = common_negatives
    
    return ", ".join(negatives)

