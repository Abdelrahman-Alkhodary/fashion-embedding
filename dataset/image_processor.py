import torchvision.transforms as transforms

# 1. Data Augmentation Transforms

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Adjust size as needed
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
])

# For the validation/test set, use simpler transforms (no random)
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. CustomDataset with Augmentations

class CustomDataset(Dataset):
    def __init__(self, image_pairs, labels, transform=None):
        self.image_pairs = image_pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Apply Transformations:
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label


# 3. Siamese Network (Modified for Data Augmentation)
class SiameseNetwork(nn.Module):
    # ... (same as before)

    def forward(self, input1, input2):
        # Apply augmentations during training only
        if self.training:
            input1 = self.train_transforms(input1)
            input2 = self.train_transforms(input2)

        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# 4. Data Loading
train_dataset = CustomDataset(train_image_pairs, train_labels, transform=train_transforms)
val_dataset = CustomDataset(val_image_pairs, val_labels, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 5. Training Loop (Instantiate model, optimizer, loss, dataloader, train)
# ... (same as before)
