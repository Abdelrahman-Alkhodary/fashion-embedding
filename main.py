import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.polyvore import PolyvoreDataset
from model.resnet import SiameseNetwork
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



def main():
    data_dir = 'E:\AlgorithmX\polyvore_outfits'
    # 1. Data Augmentation Transforms to be used in the Siamese Network model for creating positive samples
    augmented_img_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Adjust size as needed
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])

    # For the creating the anchor and negative samples, use simpler transforms 
    img_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_train_dataset = PolyvoreDataset(data_dir=data_dir, dataset_type='train', img_transforms=img_transforms, augmented_img_transforms=augmented_img_transforms, target='image')
    img_val_dataset = PolyvoreDataset(data_dir=data_dir, dataset_type='valid', img_transforms=img_transforms, augmented_img_transforms=augmented_img_transforms, target='image')
    img_test_dataset = PolyvoreDataset(data_dir=data_dir, dataset_type='test', img_transforms=img_transforms, augmented_img_transforms=augmented_img_transforms, target='image')

    img_train_loader = DataLoader(img_train_dataset, batch_size=32, shuffle=True)
    img_val_loader = DataLoader(img_val_dataset, batch_size=32, shuffle=False)
    img_test_loader = DataLoader(img_test_dataset, batch_size=32, shuffle=False)

    # 2. Siamese Network Architecture
    model = SiameseNetwork(model_name = 'resnet50', embedding_dim=128)

    # 3. Loss Function and Optimizer
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    # 4. Training Loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    training_losses = []
    validation_losses = []
    for epoch in range(10):
        model.train()
        training_loss = 0.0
        for batch in tqdm(img_train_loader, desc="Training"):
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            output1, output2, output3 = model(anchor, positive, negative)
            loss = triplet_loss(output1, output2, output3)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
        training_losses.append(training_loss/len(img_train_loader))

        # 5. Validation Loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(img_val_loader, desc="Validation"):
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

                output1, output2, output3 = model(anchor, positive, negative)
                loss = triplet_loss(output1, output2, output3)
                valid_loss += loss.item()
        validation_losses.append(valid_loss/len(img_val_loader))

        print(f"Epoch {epoch+1}, Training Loss: {training_loss/len(img_train_loader)}, Validation Loss: {valid_loss/len(img_val_loader)}")
    
    # 6. Testing Loop
    model.eval()
    anchor_pos_similarities = []
    anchor_neg_similarities = []

    with torch.no_grad():
        for batch in tqdm(img_test_loader, desc="Testing"):
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            output1, output2, output3 = model(anchor, positive, negative)
            
            # Calculate Similarities
            for i in range(output1.size(0)):  # Iterate over batch items
                anchor_pos_similarities.append(cosine_similarity(output1[i].unsqueeze(0), output2[i].unsqueeze(0)).item())
                anchor_neg_similarities.append(cosine_similarity(output1[i].unsqueeze(0), output3[i].unsqueeze(0)).item())

    # 7. Save Model Weights
    torch.save(model.state_dict(), "./weights/image_model/siamese_model_weights.pth")

    # # ... (Later, to load the weights)
    # model = SiameseNetwork(model_name = 'resnet50', embedding_dim=128)  # Recreate model
    # model.load_state_dict(torch.load("siamese_model_weights.pth"))
    # model.to(device) 
    # Analyze and Visualize
    avg_anchor_pos_sim = np.mean(anchor_pos_similarities)
    avg_anchor_neg_sim = np.mean(anchor_neg_similarities)
    print(f"Average Similarity between Anchor and Positive: {avg_anchor_pos_sim}")
    print(f"Average Similarity between Anchor and Negative: {avg_anchor_neg_sim}")


if __name__ == '__main__':
    main()


# # 2. Contrastive Loss
# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
#         return loss_contrastive


# # 8. Visualize Embeddings (Optional)
# def visualize_embeddings(embeddings, labels, method="tsne"):
#     if method == "tsne":
#         reducer = TSNE(n_components=2, random_state=42)
#     elif method == "umap":
#         reducer = umap.UMAP(n_components=2, random_state=42)
#     else:
#         raise ValueError("Invalid dimensionality reduction method")

#     reduced_embeddings = reducer.fit_transform(embeddings)

#     plt.figure(figsize=(10, 8))
#     plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap="viridis")
#     plt.colorbar()
#     plt.title(f"Fashion Item Embeddings ({method.upper()})")
#     plt.show()

# # Combine Anchor and Positive Embeddings (or any others you want to visualize)
# all_embeddings = torch.cat([output1, output2], dim=0).cpu().numpy()
# all_labels = [0] * output1.size(0) + [1] * output2.size(0)  # 0 for anchor, 1 for positive

# visualize_embeddings(all_embeddings, all_labels)  # Try both t-SNE and UMAP
