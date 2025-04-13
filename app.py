from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file
import os
import time
import uuid
from werkzeug.utils import secure_filename
import torch

from dotenv import load_dotenv

load_dotenv()
# Import the necessary functions from the backend
from ensemble import load_models, load_saved_model
from predict import predict_image
from report import export_analysis_report
from explainable_ai import generate_explanation  # Make sure to import this

# Define model paths
SIGLIP_MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
SAVE_PATH = "enhanced_ensemble_model.pth"  # Path for saved model
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'tiff'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload and reports directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Load models globally to avoid reloading for each request
print("Loading AI detection models...")
ensemble_model, siglip_model, siglip_processor = load_models()

# Load saved weights if available
try:
    ensemble_model = load_saved_model(SAVE_PATH)
    print("Saved model weights loaded successfully")
except Exception as e:
    print(f"Using base pre-trained models with equal weights. Error: {str(e)}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{str(uuid.uuid4())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        # Process the image
        try:
            start_time = time.time()
            results = predict_image(filepath, ensemble_model, siglip_model, siglip_processor)
            inference_time = time.time() - start_time

            if not results:
                return jsonify({'error': 'Image analysis failed'}), 500

            # Add explanation with inference time
            if 'explanation' not in results:
                results['explanation'] = generate_explanation(filepath, ensemble_model, results, inference_time)
            else:
                # If explanation already exists, add inference time to it
                results['explanation']['inference_time'] = inference_time

            # Generate HTML report
            report_filename = f"report_{str(uuid.uuid4())}.html"
            report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
            export_analysis_report(filepath, results, report_path)

            # Prepare results for display
            display_results = {
                'predicted_class': results['predicted_class'],
                'confidence': float(results['confidence']),
                'inference_time': inference_time,
                'model_results': {model_name:
                                      {'ai': float(probs['ai']), 'human': float(probs['human'])}
                                  for model_name, probs in results['models'].items()},
                'report_url': url_for('get_report', report_name=report_filename),
                'original_filename': file.filename,
                'image_url': url_for('get_uploaded_file', filename=unique_filename)
            }

            # Add metadata analysis if available
            if 'metadata_analysis' in results:
                display_results['metadata_analysis'] = {
                    'ai_signs': results['metadata_analysis']['ai_signs'][:3] if results['metadata_analysis'][
                        'ai_signs'] else [],
                    'human_signs': results['metadata_analysis']['human_signs'][:3] if results['metadata_analysis'][
                        'human_signs'] else []
                }

            # Add explanation if available
            if 'explanation' in results:
                display_results['explanation'] = {
                    'summary': results['explanation']['summary'],
                    'key_features': results['explanation'].get('key_features', []),  # Keep for backward compatibility
                    'inference_time': results['explanation']['inference_time']
                }

            return jsonify({'success': True, 'results': display_results})

        except Exception as e:
            return jsonify({'error': f'Analysis error: {str(e)}'}), 500

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    if 'images[]' not in request.files:
        return jsonify({'error': 'No images part in the request'}), 400

    files = request.files.getlist('images[]')

    if not files or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400

    batch_results = {
        'total': len(files),
        'processed': 0,
        'ai': 0,
        'human': 0,
        'failed': 0,
        'results': []
    }

    # Process each file
    for file in files:
        if file and allowed_file(file.filename):
            # Generate a unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{str(uuid.uuid4())}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            # Process the image
            try:
                start_time = time.time()
                results = predict_image(filepath, ensemble_model, siglip_model, siglip_processor)
                inference_time = time.time() - start_time

                if results:
                    # Update batch results
                    batch_results['processed'] += 1
                    batch_results['ai'] += 1 if results['predicted_class'] == 'ai' else 0
                    batch_results['human'] += 1 if results['predicted_class'] == 'human' else 0

                    # Add individual result
                    batch_results['results'].append({
                        'filename': file.filename,
                        'predicted_class': results['predicted_class'],
                        'confidence': float(results['confidence']),
                        'inference_time': inference_time,
                        'image_url': url_for('get_uploaded_file', filename=unique_filename)
                    })
                else:
                    batch_results['failed'] += 1
            except Exception as e:
                batch_results['failed'] += 1
                print(f"Error processing {file.filename}: {str(e)}")

    # Generate batch report HTML
    report_filename = f"batch_report_{str(uuid.uuid4())}.html"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)

    # Create batch report HTML
    generate_batch_report(batch_results, report_path)

    batch_results['report_url'] = url_for('get_report', report_name=report_filename)

    return jsonify({'success': True, 'results': batch_results})


def generate_batch_report(batch_results, output_path):
    """Generate a simple HTML report for batch processing"""
    html_content = f"""<!DOCTYPE html>
    <html>
    <head>
        <title>Batch Analysis Summary</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 0 auto; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr.ai {{ background-color: #ffebee; }}
            tr.human {{ background-color: #e8f5e9; }}
            .summary-box {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Batch Image Analysis Summary</h1>
        <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="summary-box">
            <h2>Summary</h2>
            <p>Total images analyzed: {batch_results['total']}</p>
            <p>AI-generated images: {batch_results['ai']}</p>
            <p>Human photos: {batch_results['human']}</p>
            <p>Failed analyses: {batch_results['failed']}</p>
        </div>

        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>#</th>
                <th>Filename</th>
                <th>Prediction</th>
                <th>Confidence</th>
                <th>Time (s)</th>
            </tr>
    """

    # Add rows for each image
    for idx, result in enumerate(batch_results['results']):
        html_content += f"""
            <tr class="{result['predicted_class']}">
                <td>{idx + 1}</td>
                <td>{result['filename']}</td>
                <td>{result['predicted_class'].upper()}</td>
                <td>{result['confidence']:.4f}</td>
                <td>{result.get('inference_time', 0):.3f}</td>
            </tr>
        """

    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)


@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))


@app.route('/reports/<report_name>')
def get_report(report_name):
    return send_file(os.path.join(app.config['REPORTS_FOLDER'], report_name))


@app.route('/results/<result_id>')
def show_results(result_id):
    # This would typically load results from a database
    # For simplicity, we're passing result_id to the template
    return render_template('result.html', result_id=result_id)


if __name__ == '__main__':
    app.run(debug=True)