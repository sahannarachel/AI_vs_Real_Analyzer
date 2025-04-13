import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from transformers import AutoImageProcessor, SiglipForImageClassification

# Global variable for image size
IMAGE_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the EnsembleModel class
class EnsembleModel(nn.Module):
    def __init__(self, num_classes=2):
        super(EnsembleModel, self).__init__()
        
        # Load pre-trained EfficientNetB3
        self.efficient_net = models.efficientnet_b3(weights="IMAGENET1K_V1")
        num_ftrs_efficient = self.efficient_net.classifier[1].in_features
        self.efficient_net.classifier[1] = nn.Linear(num_ftrs_efficient, num_classes)
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights="IMAGENET1K_V1")
        num_ftrs_resnet = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs_resnet, num_classes)
        
        # Ensemble weights (learnable parameters)
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)  # Initialize with equal weights
        self.softmax = nn.Softmax(dim=0)  # For normalizing weights
        
    def forward(self, x, siglip_output=None):
        # Forward pass through EfficientNet
        efficient_output = self.efficient_net(x)
        
        # Forward pass through ResNet
        resnet_output = self.resnet(x)
        
        # Get normalized ensemble weights
        weights = self.get_normalized_weights()
        
        # Initialize ensemble output
        ensemble_output = None
        
        # Handle different combinations of available model outputs
        if siglip_output is not None:
            # All models are available
            ensemble_output = (
                weights[0] * efficient_output + 
                weights[1] * resnet_output + 
                weights[2] * siglip_output
            )
        else:
            # Only base models are available
            adjusted_weights = weights[:2] / weights[:2].sum()
            ensemble_output = (
                adjusted_weights[0] * efficient_output + 
                adjusted_weights[1] * resnet_output
            )
        
        return ensemble_output
    
    def get_normalized_weights(self):
        return self.softmax(self.ensemble_weights)

# Define transforms
def get_transforms():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load models function
def load_models(SIGLIP_MODEL_ID="Ateeqq/ai-vs-human-image-detector"):
    # Load ensemble model
    ensemble_model = EnsembleModel().to(device)
    
    # Load SiglipModel for AI vs Human detection
    siglip_processor = AutoImageProcessor.from_pretrained(SIGLIP_MODEL_ID)
    siglip_model = SiglipForImageClassification.from_pretrained(SIGLIP_MODEL_ID).to(device)
    
    return ensemble_model, siglip_model, siglip_processor

# Load saved model function
def load_saved_model(path, SAVE_PATH="enhanced_ensemble_model.pth"):
    ensemble_model = EnsembleModel().to(device)
    
    try:
        checkpoint = torch.load(path, map_location=device)
        ensemble_model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Model loaded from {path}")
        weights = ensemble_model.get_normalized_weights().cpu().detach().numpy()
        print(f"Ensemble weights: EfficientNet={weights[0]:.4f}, ResNet={weights[1]:.4f}, "
              f"Siglip={weights[2]:.4f}")
    except Exception as e:
        print(f"Using base pre-trained models with equal weights. Error: {e}")
    
    return ensemble_model