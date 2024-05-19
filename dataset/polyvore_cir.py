# -*- coding:utf-8 -*-

import os
import math
import json
import datetime
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from numpy import ndarray

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2


@dataclass
class DatasetArguments:
    polyvore_split: str = 'nondisjoint'
    dataset_type: str = 'train'
    image_size: int = 224
    image_transform: Optional[List[Any]] = None
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    

class PolyvoreDataset_cir(Dataset):
    def __init__(
            self,
            data_dir: str,
            polyvore_split: str = 'nondisjoint',
            dataset_type: str = 'train',
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            tokenizer: Optional[AutoTokenizer] = None
            ):

        self.is_train = (dataset_type == 'train')
        # Meta Data preprocessing
        self.item_ids, self.item_id2idx, \
        self.item_id2category, self.category2item_ids, self.categories, \
            self.outfit_id2item_id, self.item_id2desc = load_data(data_dir, polyvore_split, dataset_type)
        
        # Data Configurations
        self.img_dir = os.path.join(data_dir, 'images')
        # print(f'Use custom transform: {use_custom_transform}')
        self.image_processor = DeepFashionImageProcessor(size=args.image_size, use_custom_transform=use_custom_transform, custom_transform=args.image_transform)
        self.use_text = args.use_text
        if args.use_text:
            # print(f'Use text: {args.use_text}')
            self.input_processor = DeepFashionInputProcessor(
                categories=self.categories, use_image=True, image_processor=self.image_processor, \
                use_text=True, text_tokenizer=tokenizer, text_max_length=args.text_max_length, outfit_max_length=args.outfit_max_length, device=args.device)
            
        # Input
        self.data = load_data_inputs(data_dir, polyvore_split, dataset_type, self.outfit_id2item_id)
        
    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_txt(self, item_id):
        desc = self.item_id2desc[item_id] if item_id in self.item_id2desc else self.item_id2category[item_id]
        return desc
    
    def _get_inputs(self, item_id) -> Dict[Literal['text', 'image'], Tensor]:
        image = self._load_img(item_id)
        text = self._load_txt(item_id)
        return self.input_processor(image, text)

    def __getitem__(self, idx):   
        sample = self.data[idx]
        item_1_id = sample['item_1']
        item_2_id = sample['item_2']
        item_1 = self._get_inputs(item_1_id)
        item_2 = self._get_inputs(item_2_id)
        return  {'item_1': item_1, 'item_2': item_2}
        
    def __len__(self):
        return len(self.data)
    

def load_data_inputs(data_dir, polyvore_split, dataset_type, outfit_id2item_id):
    """To make sure the items are different we will use fill-in-the-blank data from polyvore dataset 
    by sampling one item from question and one from answers from the same outfit."""
    fitb_path = os.path.join(data_dir, polyvore_split, f'fill_in_blank_{dataset_type}.json')
    with open(fitb_path, 'r') as f:
        fitb_data = json.load(f)
        data = []
        for item in fitb_data:
            question_ids = list(map(lambda x: outfit_id2item_id[x], item['question']))
            answers_ids = list(map(lambda x: outfit_id2item_id[x], item['answers'])) # all answers for evaluation
            for id in question_ids:
                data.append({
                    'item_1': id,
                    'item_2': np.random.choice(answers_ids)
                })
    return data


def load_data(data_dir, polyvore_split, dataset_type):
    # Paths
    # data_dir = os.path.join(data_dir, args.polyvore_split)
    outfit_data_path = os.path.join(data_dir, polyvore_split, f'{dataset_type}.json')
    meta_data_path = os.path.join(data_dir, 'polyvore_item_metadata.json')
    outfit_data = json.load(open(outfit_data_path))
    meta_data = json.load(open(meta_data_path))
    # Load
    item_ids = set()
    categories = set()
    item_id2category = {}
    item_id2desc = {}
    category2item_ids = {}
    outfit_id2item_id = {}
    for outfit in outfit_data:
        outfit_id = outfit['set_id']
        for item in outfit['items']:
            # Item of cloth
            item_id = item['item_id']
            item_ids.add(item_id)
            # Category of cloth
            category = meta_data[item_id]['semantic_category']
            categories.add(category)
            item_id2category[item_id] = category
            if category not in category2item_ids:
                category2item_ids[category] = set()
            category2item_ids[category].add(item_id)
            # Description of cloth
            desc = meta_data[item_id]['title']
            if not desc:
                desc = meta_data[item_id]['url_name']
            item_id2desc[item_id] = desc.replace('\n','').strip().lower()
            # Replace the item code with the outfit number with the image code
            outfit_id2item_id[f"{outfit['set_id']}_{item['index']}"] = item_id
            
    item_ids = list(item_ids)
    item_id2idx = {id : idx for idx, id in enumerate(item_ids)}
    categories = list(categories)

    return item_ids, item_id2idx, \
        item_id2category, category2item_ids, categories, \
            outfit_id2item_id, item_id2desc