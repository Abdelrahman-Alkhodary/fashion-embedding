import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.losses import ContrastiveTensionDataLoader
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.polyvore_text import text_data_list
import numpy as np



def main():
    data_dir = 'E:\AlgorithmX\polyvore_outfits'
    # 1. Dataset using Text Data
    text_train_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint', dataset_type='train')
    text_val_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint',dataset_type='valid')
    text_test_dataset = text_data_list(data_dir=data_dir, polyvore_split='nondisjoint',dataset_type='test')

    train_dataloader = ContrastiveTensionDataLoader(text_train_dataset, batch_size=3, pos_neg_ratio=3)
    
    # 2. Sentence Transformer Model
    word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(
        in_features=pooling_model.get_sentence_embedding_dimension(),
        out_features=128,
        activation_function=nn.Tanh(),
    )

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], cache_folder='./weights/text_model/cache', output_path='./weights/text_model' )

    # 3. Loss Function and Optimizer
    train_loss = losses.ContrastiveTensionLoss(model=model)


    # 4. Training Loop
    model.fit(
        [(train_dataloader, train_loss)],
        epochs=10,
        checkpoint_path='./weights/text_model'
    )
    

if __name__ == '__main__':
    main()