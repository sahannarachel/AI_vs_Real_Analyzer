import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ensemble import load_models, device

# Function to train the ensemble weights if needed
def train_ensemble_weights(train_loader, ensemble_model, siglip_model, siglip_processor, num_epochs=5):
    """
    Optional function to train the ensemble weights based on a dataset
    """
    # Freeze base model parameters to only update ensemble weights
    for param in ensemble_model.parameters():
        param.requires_grad = False
    
    # Only enable gradient for ensemble weights
    ensemble_model.ensemble_weights.requires_grad = True
    
    optimizer = torch.optim.Adam([ensemble_model.ensemble_weights], lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get outputs from Siglip model
            siglip_inputs = siglip_processor(images=[Image.fromarray(np.uint8(img.permute(1, 2, 0).cpu().numpy()*255)) 
                                               for img in inputs], return_tensors="pt").to(device)
            
            # Forward pass through Siglip model
            with torch.no_grad():
                siglip_output = siglip_model(**siglip_inputs).logits
            
            # Forward pass through ensemble
            optimizer.zero_grad()
            outputs = ensemble_model(inputs, siglip_output)
            
            # Calculate loss and update weights
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Print epoch statistics
        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Accuracy={accuracy:.2f}%')
        print(f'Ensemble weights: {ensemble_model.get_normalized_weights().cpu().detach().numpy()}')
    
    print("Training completed!")
    return ensemble_model