import time
import base64
import io
import numpy as np
import cv2
from PIL import Image

# Function to export full analysis report
def export_analysis_report(image_path, results, output_path="analysis_report.html"):
    """
    Export a detailed analysis report as HTML
    """
    try:
        # Convert image to base64 for embedding in HTML
        with open(image_path, "rb") as img_file:
            import base64
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Generate heatmap image in base64 if available
        heatmap_data = None
        if 'heatmaps' in results and results['heatmaps'] is not None:
            # Get original image
            image = Image.open(image_path).convert('RGB')
            img_np = np.array(image)
            
            # Get combined heatmap
            heatmap = results['heatmaps']['combined']
            
            # Convert heatmap to colormap
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            img_resized = cv2.resize(img_np, (heatmap_colored.shape[1], heatmap_colored.shape[0]))
            overlay = cv2.addWeighted(img_resized, 0.7, heatmap_colored, 0.3, 0)
            
            # Convert to PIL and then to base64
            overlay_pil = Image.fromarray(overlay)
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format="PNG")
            heatmap_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Create HTML content
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>AI vs Human Image Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; text-align: center; margin-bottom: 20px; }}
                .result-box {{ background-color: {'#ffebee' if results['predicted_class'] == 'ai' else '#e8f5e9'}; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .result-title {{ font-size: 24px; font-weight: bold; }}
                .confidence {{ font-size: 18px; }}
                .section {{ margin-bottom: 30px; }}
                .section-title {{ border-bottom: 1px solid #ddd; padding-bottom: 10px; }}
                .image-container {{ display: flex; justify-content: space-between; margin-bottom: 20px; }}
                .image-box {{ width: 48%; }}
                .model-results {{ display: flex; flex-wrap: wrap; }}
                .model-card {{ width: 23%; margin-right: 2%; margin-bottom: 20px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .metadata-list {{ list-style-type: none; padding-left: 0; }}
                .metadata-item {{ padding: 8px; margin-bottom: 5px; background-color: #f9f9f9; }}
                .metadata-item.ai {{ background-color: #ffebee; }}
                .metadata-item.human {{ background-color: #e8f5e9; }}
                .explanation {{ background-color: #fff8e1; padding: 15px; border-left: 4px solid #ffc107; }}
                progress {{ width: 100%; height: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI vs Human Image Analysis Report</h1>
                <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="result-box">
                <div class="result-title">Verdict: This image appears to be {results['predicted_class'].upper()}-GENERATED</div>
                <div class="confidence">Overall confidence: {results['confidence']:.2f}</div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Image Analysis</h2>
                <div class="image-container">
                    <div class="image-box">
                        <h3>Original Image</h3>
                        <img src="data:image/jpeg;base64,{img_data}" style="max-width: 100%; max-height: 400px;">
                    </div>
        """
        
        # Add heatmap if available
        if heatmap_data:
            html_content += f"""
                    <div class="image-box">
                        <h3>AI Artifacts Heatmap</h3>
                        <img src="data:image/png;base64,{heatmap_data}" style="max-width: 100%; max-height: 400px;">
                        <p><em>Heatmap shows regions that contribute most to AI classification (red = strongest indicators)</em></p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Model Predictions</h2>
                <div class="model-results">
        """
        
        # Add model cards
        models = ['efficientnet', 'resnet', 'siglip', 'ensemble']
        for model in models:
            ai_prob = results['models'][model]['ai']
            human_prob = results['models'][model]['human']
            prediction = 'AI' if ai_prob > human_prob else 'HUMAN'
            html_content += f"""
                    <div class="model-card">
                        <h3>{model.upper()}</h3>
                        <p>Prediction: <strong>{prediction}</strong></p>
                        <p>AI probability:</p>
                        <progress value="{ai_prob}" max="1"></progress>
                        <p>{ai_prob:.4f}</p>
                        <p>Human probability:</p>
                        <progress value="{human_prob}" max="1"></progress>
                        <p>{human_prob:.4f}</p>
                    </div>
            """
        
        html_content += """
                </div>
            </div>
        """
        
        # Add explanation section
        if 'explanation' in results:
            explanation = results['explanation']
            html_content += f"""
            <div class="section">
                <h2 class="section-title">Analysis Explanation</h2>
                <div class="explanation">
                    {explanation['summary'].replace('\n', '<br>')}
                </div>
                
                <h3>Key Features Detected</h3>
                <ul>
            """
            
            # Add key features
            for feature in explanation.get('key_features', []):
                html_content += f"<li>{feature}</li>"
            
            html_content += """
                </ul>
                
                <h3>Technical Model Analysis</h3>
                <ul>
            """
            
            # Add technical details
            for model, details in explanation.get('technical_details', {}).items():
                html_content += f"<li>{details['decision']}</li>"
            
            html_content += """
                </ul>
            </div>
            """
        
        # Add metadata section
        if 'metadata_analysis' in results:
            metadata = results['metadata_analysis']
            html_content += """
            <div class="section">
                <h2 class="section-title">Metadata Analysis</h2>
            """
            
            # Add metadata score if available
            if 'metadata_score' in metadata:
                ai_likelihood = metadata['metadata_score']['ai_likelihood']
                human_likelihood = metadata['metadata_score']['human_likelihood']
                html_content += f"""
                <div style="margin-bottom: 20px;">
                    <h3>Metadata-based likelihood</h3>
                    <p>AI indicators strength:</p>
                    <progress value="{ai_likelihood}" max="1"></progress>
                    <p>{ai_likelihood:.2f}</p>
                    <p>Human indicators strength:</p>
                    <progress value="{human_likelihood}" max="1"></progress>
                    <p>{human_likelihood:.2f}</p>
                </div>
                """
            
            # Add AI signs
            if metadata['ai_signs']:
                html_content += """
                <h3>AI Indicators Found</h3>
                <ul class="metadata-list">
                """
                for sign in metadata['ai_signs']:
                    html_content += f'<li class="metadata-item ai">{sign}</li>'
                html_content += "</ul>"
            
            # Add human signs
            if metadata['human_signs']:
                html_content += """
                <h3>Human Indicators Found</h3>
                <ul class="metadata-list">
                """
                for sign in metadata['human_signs']:
                    html_content += f'<li class="metadata-item human">{sign}</li>'
                html_content += "</ul>"
            
            # Add raw metadata if available
            if metadata['raw_metadata']:
                html_content += """
                <h3>Raw Metadata</h3>
                <div style="max-height: 300px; overflow-y: auto; background-color: #f8f8f8; padding: 15px; font-family: monospace;">
                """
                for key, value in metadata['raw_metadata'].items():
                    html_content += f"<p><strong>{key}:</strong> {value}</p>"
                html_content += "</div>"
            
            html_content += """
            </div>
            """
        
        # Close HTML
        html_content += """
            <div class="section">
                <p style="text-align: center; color: #666;">This analysis is provided for informational purposes only.</p>
            </div>
        </body>
        </html>
        """
        
        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)
        
        print(f"Analysis report exported to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error exporting report: {e}")
        return False