import torch
import numpy as np
from PIL import Image
from ensemble import get_transforms, device
from analyze_metadata import analyze_metadata
from heatmap import generate_heatmap
from explainable_ai import generate_explanation

def predict_image(image_path, ensemble_model, siglip_model, siglip_processor):
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    
    # Transform for EfficientNet and ResNet
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get EfficientNet prediction
    ensemble_model.efficient_net.eval()
    with torch.no_grad():
        efficient_output = ensemble_model.efficient_net(img_tensor)
        efficient_probs = torch.softmax(efficient_output, dim=1).cpu().numpy()[0]
    
    # Get ResNet prediction
    ensemble_model.resnet.eval()
    with torch.no_grad():
        resnet_output = ensemble_model.resnet(img_tensor)
        resnet_probs = torch.softmax(resnet_output, dim=1).cpu().numpy()[0]
    
    # Get Siglip prediction
    siglip_model.eval()
    with torch.no_grad():
        siglip_inputs = siglip_processor(images=image, return_tensors="pt").to(device)
        siglip_output = siglip_model(**siglip_inputs)
        siglip_probs = torch.softmax(siglip_output.logits, dim=1).cpu().numpy()[0]
    
    # Get normalized ensemble weights
    weights = ensemble_model.get_normalized_weights().cpu().detach().numpy()
    
    # Combine predictions with weighted average
    ensemble_probs = (
        weights[0] * efficient_probs + 
        weights[1] * resnet_probs + 
        weights[2] * siglip_probs
    )
    
    # Get the predicted class (0 for AI, 1 for Human)
    class_names = ['ai', 'human']
    predicted_class_idx = np.argmax(ensemble_probs)
    predicted_class = class_names[predicted_class_idx]
    confidence = ensemble_probs[predicted_class_idx]
    
    # Analyze image metadata
    metadata_analysis = analyze_metadata(image_path)
    
    # Generate manipulation heatmap
    heatmaps = generate_heatmap(image_path, ensemble_model)
    
    # Return results
    results = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'models': {
            'efficientnet': {'ai': efficient_probs[0], 'human': efficient_probs[1]},
            'resnet': {'ai': resnet_probs[0], 'human': resnet_probs[1]},
            'siglip': {'ai': siglip_probs[0], 'human': siglip_probs[1]},
            'ensemble': {'ai': ensemble_probs[0], 'human': ensemble_probs[1]}
        },
        'metadata_analysis': metadata_analysis,
        'heatmaps': heatmaps
    }
    
    # Generate explanation for the prediction
    explanation = generate_explanation(image_path, ensemble_model, results)
    results['explanation'] = explanation
    
    return results