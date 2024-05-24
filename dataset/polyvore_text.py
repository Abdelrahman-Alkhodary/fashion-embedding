import os
import json
from tqdm import tqdm
import random
from sentence_transformers.readers import InputExample


# def text_data_list(data_dir, polyvore_split, dataset_type, samples_per_category=20):
#     item_ids, item_id2category, item_id2desc = load_data(data_dir, polyvore_split, dataset_type)
#     # create a dictionary to store the sentences of each category
#     categorized_sentences = {}
#     for item_id in tqdm(item_ids):
#         item_desc = item_id2desc[item_id] if item_id in item_id2desc else item_id2category[item_id]
#         item_category = item_id2category[item_id]
#         categorized_sentences.setdefault(item_category, []).append(item_desc)
    
#     sentece_pairs = []
#     categories = list(categorized_sentences.keys())
#     for cat_1 in categories:
#         for cat_2 in categories:
#             if cat_1 != cat_2:
#                 # random sampling pairs of sentences from different categories
#                 sentences_1 = random.sample(categorized_sentences[cat_1], samples_per_category)
#                 sentences_2 = random.sample(categorized_sentences[cat_2], samples_per_category)
                
#                 for sent_1 in sentences_1:
#                     for sent_2 in sentences_2:
#                         sentece_pairs.append((sent_1, sent_2))
#     return sentece_pairs
cat2label = {'accessories': 0, 
             'all-body': 1, 
             'bags': 2, 
             'bottoms': 3, 
             'hats': 4, 
             'jewellery': 5, 
             'outerwear': 6, 
             'scarves': 7, 
             'shoes': 8, 
             'sunglasses': 9, 
             'tops': 10
}
        

def text_data_list(data_dir, polyvore_split, dataset_type):
    item_ids, item_id2category, item_id2desc = load_data(data_dir, polyvore_split, dataset_type)
    def _load_txt(item_id):
        desc = item_id2desc[item_id] if item_id in item_id2desc else item_id2category[item_id]
        text = item_id2category[item_id] + ' ' + desc
        return text
    data_list = []
    for item_id in tqdm(item_ids):
        data_list.append(InputExample(texts=[_load_txt(item_id)], label=cat2label[item_id2category[item_id]]))
    return data_list

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

    return item_ids, item_id2category, item_id2desc