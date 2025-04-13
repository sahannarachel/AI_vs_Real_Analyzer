import torch
import numpy as np
import cv2
from PIL import Image
from captum.attr import LayerGradCam
from ensemble import get_transforms, device

def generate_heatmap(image_path, ensemble_model):
    """
    Generate a heatmap highlighting regions that contribute to AI classification
    """
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        transform = get_transforms()
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Create a visualization for both ResNet and EfficientNet models
        heatmaps = {}
        
        # 1. For ResNet
        resnet = ensemble_model.resnet
        resnet.eval()
        
        # Target layer for GradCAM (usually the final convolutional layer)
        target_layer = resnet.layer4[-1]
        
        # Create GradCAM object
        grad_cam = LayerGradCam(resnet, target_layer)
        
        # Generate attribution (heatmap)
        attribution_resnet = grad_cam.attribute(img_tensor, target=0)  # 0 is AI class
        
        # Normalize and convert to heatmap
        attribution_resnet = attribution_resnet.detach().cpu().numpy()
        heatmap_resnet = np.mean(attribution_resnet, axis=1)[0]
        heatmap_resnet = np.maximum(heatmap_resnet, 0)
        heatmap_resnet = heatmap_resnet / (np.max(heatmap_resnet) + 1e-10)
        
        # Store ResNet heatmap
        heatmaps['resnet'] = cv2.resize(heatmap_resnet, (image.width, image.height))
        
        # 2. For EfficientNet
        efficientnet = ensemble_model.efficient_net
        efficientnet.eval()
        
        # Target layer for EfficientNet (features module's last layer)
        target_layer = efficientnet.features[-1]
        
        # Create GradCAM object
        grad_cam = LayerGradCam(efficientnet, target_layer)
        
        # Generate attribution (heatmap)
        attribution_efficient = grad_cam.attribute(img_tensor, target=0)  # 0 is AI class
        
        # Normalize and convert to heatmap
        attribution_efficient = attribution_efficient.detach().cpu().numpy()
        heatmap_efficient = np.mean(attribution_efficient, axis=1)[0]
        heatmap_efficient = np.maximum(heatmap_efficient, 0)
        heatmap_efficient = heatmap_efficient / (np.max(heatmap_efficient) + 1e-10)
        
        # Store EfficientNet heatmap
        heatmaps['efficientnet'] = cv2.resize(heatmap_efficient, (image.width, image.height))
        
        # 3. Combined heatmap (weighted average based on model weights)
        weights = ensemble_model.get_normalized_weights().cpu().detach().numpy()
        
        # Rescale both heatmaps to the same size
        heatmap_resnet_resized = cv2.resize(heatmap_resnet, (image.width, image.height))
        heatmap_efficient_resized = cv2.resize(heatmap_efficient, (image.width, image.height))
        
        # Calculate weighted average
        combined_heatmap = (weights[0] * heatmap_efficient_resized + 
                            weights[1] * heatmap_resnet_resized)
        
        # Normalize
        combined_heatmap = np.maximum(combined_heatmap, 0)
        combined_heatmap = combined_heatmap / (np.max(combined_heatmap) + 1e-10)
        
        # Store combined heatmap
        heatmaps['combined'] = combined_heatmap
        
        return heatmaps
        
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None