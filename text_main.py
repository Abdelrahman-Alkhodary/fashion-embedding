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
    text_val_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint',dataset_type='valid')
    text_test_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint',dataset_type='test')

    # train_dataloader = ContrastiveTensionDataLoader(text_train_dataset, batch_size=10, pos_neg_ratio=10)
    
    # Model for which we apply dimensionality reduction
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # New size for the embeddings
    new_dimension = 128
    
    # get the embedding for the dataset
    train_embeddings = model.encode(text_train_dataset, convert_to_numpy=True)
    
    # Compute PCA on the train embeddings matrix
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)
    
    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    dense = models.Dense(
        in_features=model.get_sentence_embedding_dimension(),
        out_features=new_dimension,
        bias=False,
        activation_function=torch.nn.Identity(),
    )
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module("dense", dense)
    
    # If you like, you can store the model on disc by uncommenting the following line
    model.save('./weights/text_model/sbert-128dim-model')
    
    # finished
    print("Model is stored on disc")
    
    # You can then load the adapted model that produces 128 dimensional embeddings like this:
    # model = SentenceTransformer('models/my-128dim-model')
    

if __name__ == '__main__':
    main()