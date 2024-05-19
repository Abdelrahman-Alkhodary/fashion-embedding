import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50, swin_t  
from torch.hub import load_state_dict_from_url
from tqdm import tqdm

# 1. Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self, model_name, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Load ResNet or Swin Transformer
        model_name = "resnet50"  # or "swin_t"
        if model_name == "resnet50":
            self.backbone = resnet50(pretrained=False)
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                map_location=torch.device('cuda'),
                model_dir="."  # Download to the current directory
            )
            self.backbone.load_state_dict(state_dict)

        # elif model_name == "swin_t":
        #    # Swin Transformer loading (similar approach)
        #    # Get the appropriate URL and directory from PyTorch documentation

        # Choose ResNet or Swin Transformer (uncomment one)
        self.backbone = resnet50(pretrained=False)
        # self.backbone = swin_t(weights="DEFAULT") 

        # Remove the last fully connected layer
        self.backbone.fc = nn.Identity()

        # Projection Head for Embedding
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet output size
            # nn.Linear(768, 512),  # Swin-T output size
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x):
        """The forward_once function ensures that each input passes through the exact same set of layers and transformations."""
        x = self.backbone(x)
        x = self.projection_head(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2



class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()

        # Load ResNet or Swin Transformer
        model_name = "resnet50"  # or "swin_t"
        if model_name == "resnet50":
            self.backbone = resnet50(pretrained=False)
            state_dict = load_state_dict_from_url(
                "https://download.pytorch.org/models/resnet50-0676ba61.pth",
                map_location=torch.device('cpu'),
                model_dir="."  # Download to the current directory
            )
            self.backbone.load_state_dict(state_dict)

        # elif model_name == "swin_t":
        #    # Swin Transformer loading (similar approach)
        #    # Get the appropriate URL and directory from PyTorch documentation

        # Remove the last fully connected layer
        self.backbone.fc = nn.Identity()

        # ... (rest of your SiameseNetwork code)