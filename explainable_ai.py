import numpy as np
from PIL import Image
import cv2
import torch
from ensemble import get_transforms, device


# NEW FEATURE 3: Explainable AI function
def generate_explanation(image_path, ensemble_model, results, inference_time=None):
    """
    Generate explanations for why the image was classified as AI or human
    """
    explanation = {
        'summary': '',
        'key_features': [],  # Keep this empty array to prevent frontend errors
        'technical_details': {},
        'explanation_factors': {},
        'inference_time': inference_time
    }

    # Load the image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Common AI image artifacts to check for
    ai_artifacts = {
        'symmetric_features': check_feature_symmetry(image),
        'unusual_textures': check_texture_consistency(image),
        'edge_artifacts': check_edge_artifacts(image),
        'metadata_analysis': results.get('metadata_analysis', {})
    }

    # Add artifact findings to explanation
    for feature, details in ai_artifacts.items():
        if isinstance(details, dict) and details.get('detected'):
            explanation['explanation_factors'][feature] = details.get('score', 0.5)

    # Add model confidence information
    model_explanations = {
        'efficientnet': explain_model_decision('EfficientNet',
                                               results['models']['efficientnet']['ai'],
                                               results['models']['efficientnet']['human']),
        'resnet': explain_model_decision('ResNet',
                                         results['models']['resnet']['ai'],
                                         results['models']['resnet']['human']),
        'siglip': explain_model_decision('Siglip',
                                         results['models']['siglip']['ai'],
                                         results['models']['siglip']['human'])
    }

    explanation['technical_details'] = model_explanations

    # Generate summary based on predicted class
    if results['predicted_class'] == 'ai':
        summary_parts = ["This image appears to be AI-generated because:"]

        # Add artifact findings to summary
        for feature, details in ai_artifacts.items():
            if isinstance(details, dict) and details.get('detected'):
                summary_parts.append(f"- {feature.replace('_', ' ').title()}: {details.get('details', '')}")

        # Add model decision info
        decisive_models = [model for model, vals in results['models'].items()
                           if vals['ai'] > vals['human'] and vals['ai'] > 0.6]

        if decisive_models:
            model_str = ', '.join([m.upper() for m in decisive_models])
            summary_parts.append(f"- {model_str} confidently classified this as AI-generated")

        # Add metadata info if available
        if 'metadata_analysis' in results and results['metadata_analysis']['ai_signs']:
            top_metadata_sign = results['metadata_analysis']['ai_signs'][0]
            summary_parts.append(f"- Metadata indicator: {top_metadata_sign}")

        explanation['summary'] = '\n'.join(summary_parts)
    else:
        # For human images
        summary_parts = ["This image appears to be a genuine human photo because:"]

        # Add metadata evidence if available
        if 'metadata_analysis' in results and results['metadata_analysis']['human_signs']:
            top_human_sign = results['metadata_analysis']['human_signs'][0]
            summary_parts.append(f"- Metadata indicator: {top_human_sign}")

        # Add natural feature evidence
        natural_features = [feat for feat, val in ai_artifacts.items()
                            if isinstance(val, dict) and not val.get('detected', True)]

        if natural_features:
            natural_str = ', '.join([f.replace('_', ' ') for f in natural_features])
            summary_parts.append(f"- Natural characteristics: {natural_str}")

        # Add model decision info
        decisive_models = [model for model, vals in results['models'].items()
                           if vals['human'] > vals['ai'] and vals['human'] > 0.6]

        if decisive_models:
            model_str = ', '.join([m.upper() for m in decisive_models])
            summary_parts.append(f"- {model_str} confidently classified this as human-generated")

        explanation['summary'] = '\n'.join(summary_parts)

    return explanation


def explain_model_decision(model_name, ai_prob, human_prob):
    """Helper function to explain individual model decisions"""
    confidence = max(ai_prob, human_prob)
    prediction = "AI" if ai_prob > human_prob else "Human"

    decision_factors = []

    if confidence > 0.9:
        confidence_level = "very high confidence"
    elif confidence > 0.75:
        confidence_level = "high confidence"
    elif confidence > 0.6:
        confidence_level = "moderate confidence"
    else:
        confidence_level = "low confidence"

    decision = f"{model_name} predicts {prediction} with {confidence_level} ({confidence:.2f})"

    return {
        "decision": decision,
        "confidence": confidence,
        "prediction": prediction,
        "factors": decision_factors
    }


def check_feature_symmetry(image):
    """Check for unnatural symmetry in facial features common in AI-generated images"""
    try:
        img_np = np.array(image)
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Load a face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            # For simplicity, check the first detected face
            x, y, w, h = faces[0]
            face = img_np[y:y + h, x:x + w]

            # Check left-right symmetry (too perfect symmetry is suspicious)
            left_half = face[:, :face.shape[1] // 2]
            right_half = face[:, face.shape[1] // 2:]
            right_half_flipped = cv2.flip(right_half, 1)

            # Resize if needed
            if left_half.shape != right_half_flipped.shape:
                min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                left_half = left_half[:, :min_width]
                right_half_flipped = right_half_flipped[:, :min_width]

            # Calculate difference
            diff = cv2.absdiff(left_half, right_half_flipped)
            diff_score = np.mean(diff)

            # Very low difference suggests unnatural symmetry
            symmetry_detected = diff_score < 10

            return {
                'detected': symmetry_detected,
                'score': 1.0 - (diff_score / 50),  # Normalize score
                'details': "Unnatural facial symmetry detected" if symmetry_detected else "Natural facial asymmetry"
            }

        return {'detected': False, 'score': 0, 'details': "No faces detected for symmetry analysis"}

    except Exception as e:
        return {'detected': False, 'score': 0, 'details': f"Symmetry check error: {str(e)}"}


def check_texture_consistency(image):
    """Check for unusual texture patterns common in AI-generated images"""
    try:
        img_np = np.array(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Calculate texture using Haralick features
        glcm = cv2.resize(gray, (64, 64))  # Resize for faster processing
        h_features = []

        # Calculate GLCM at different angles
        for angle in [0, 45, 90, 135]:
            glcm_mat = cv2.getGaussianKernel(5, 1.0)
            glcm_mat = glcm_mat * glcm_mat.transpose()
            glcm_filtered = cv2.filter2D(glcm, -1, glcm_mat)
            h_features.append(np.std(glcm_filtered))

        # Calculate texture consistency score
        texture_std = np.std(h_features)

        # AI images often have unusual consistency in texture patterns
        unusual_texture = texture_std < 0.5

        return {
            'detected': unusual_texture,
            'score': 1.0 - texture_std if texture_std < 1.0 else 0.0,
            'details': "Unusual texture consistency" if unusual_texture else "Natural texture variations"
        }

    except Exception as e:
        return {'detected': False, 'score': 0, 'details': f"Texture analysis error: {str(e)}"}


def check_edge_artifacts(image):
    """Check for edge artifacts common in AI-generated images"""
    try:
        img_np = np.array(image)
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 100, 200)

        # Check for unusual edge patterns
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])

        # AI images often have unusual edge patterns/densities
        unusual_edges = edge_density > 0.2 or edge_density < 0.01

        return {
            'detected': unusual_edges,
            'score': abs(edge_density - 0.1) * 5 if unusual_edges else 0,
            'details': "Unusual edge patterns detected" if unusual_edges else "Natural edge distribution"
        }

    except Exception as e:
        return {'detected': False, 'score': 0, 'details': f"Edge analysis error: {str(e)}"}