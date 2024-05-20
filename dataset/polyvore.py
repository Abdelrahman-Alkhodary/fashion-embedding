# -*- coding:utf-8 -*-

import os
import json
from tqdm import tqdm
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import cv2
from PIL import Image



class PolyvoreDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            dataset_type: str,
            target, 
            img_transforms=None,
            augmented_img_transforms=None, 
            polyvore_split: str = 'nondisjoint',
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
            ):

        self.target = target
        # Meta Data preprocessing
        self.item_ids, self.item_id2idx, \
        self.item_id2category, self.category2item_ids, self.categories, \
            self.outfit_id2item_id, self.item_id2desc = load_data(data_dir, polyvore_split, dataset_type)
        
        # Data Configurations
        self.img_dir = os.path.join(data_dir, 'images')
        self.img_transforms = img_transforms
        self.augmented_img_transforms = augmented_img_transforms
        # Input
        self.data = load_data_inputs(data_dir, polyvore_split, dataset_type, self.outfit_id2item_id)
        
    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}.jpg")
        img = Image.open(path)
        return img
    
    def _load_txt(self, item_id):
        desc = self.item_id2desc[item_id] if item_id in self.item_id2desc else self.item_id2category[item_id]
        return desc
    
    def text_data_list(self):
        data_list = []
        for item_id in tqdm(self.item_ids):
            data_list.append(self._load_txt(item_id))
    
    def _get_inputs(self, item_id, image_processor) -> Dict[Literal['text', 'image'], Tensor]:
        if self.target == 'image':
            image = self._load_img(item_id)
            transformer_img = image_processor(image)
            return transformer_img
        elif self.target == 'text':
            return self._load_txt(item_id)

    def __getitem__(self, idx):   
        sample = self.data[idx]
        item_1_id = sample['item_1']
        item_2_id = sample['item_2']
        item_1 = self._get_inputs(item_1_id, image_processor=self.img_transforms)
        item_1_augemented = self._get_inputs(item_1_id, image_processor=self.augmented_img_transforms)
        item_2 = self._get_inputs(item_2_id, image_processor=self.img_transforms)
        return  item_1, item_1_augemented, item_2
        
    def __len__(self):
        return len(self.data)


def load_data_inputs(data_dir, polyvore_split, dataset_type, outfit_id2item_id):
    """To make sure the items are different we will use fill-in-the-blank data from polyvore dataset 
    by sampling one item from question and one from answers from the same outfit."""
    fitb_path = os.path.join(data_dir, polyvore_split, f'fill_in_blank_{dataset_type}.json')
    with open(fitb_path, 'r') as f:
        fitb_data = json.load(f)
        data = []
        for item in tqdm(fitb_data):
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