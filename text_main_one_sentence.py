import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.losses import ContrastiveTensionDataLoader
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.polyvore_text import text_data_list
import numpy as np
from sklearn.decomposition import PCA



def main():
    data_dir = '/home/abdelrahman/fashion-matching/data/polyvore_outfits'
    # 1. Dataset using Text Data
    text_train_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint', dataset_type='train')
    
    # Model for which we apply dimensionality reduction
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # New size for the embeddings
    new_dimension = 128
    
    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=new_dimension,
        activation_function=nn.Tanh(),
    )
    model.add_module("dense", dense)
    
    train_batch_size = 2
    train_dataloader = DataLoader(text_train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.BatchAllTripletLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=10,
    )
    
    # If you like, you can store the model on disc by uncommenting the following line
    model.save('./weights/text_model/one_snet_sbert-128dim-model')
    
    # finished
    print("Model is stored on disc")
    
    # You can then load the adapted model that produces 128 dimensional embeddings like this:
    # model = SentenceTransformer('./weights/text_model/one_snet_sbert-128dim-model')
    

if __name__ == '__main__':
    main()