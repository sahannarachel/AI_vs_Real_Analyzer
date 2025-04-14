# Real or Rendered? AI-Generated Image Detection Web App

This project is a Flask-based interactive web application that detects whether an image is AI-generated or a real photograph using an **ensemble of deep learning models**. It also provides **explanations**, **metadata insights**, and **detailed reports** for individual and batch image analysis.

## Features

- Detect AI-generated vs Human-captured images
- Uses an **Ensemble of models** including SigLIP + CNN/ResNet/Vision Transformers
- Generates **HTML reports** for each analysis (single and batch)
- Metadata inspection (EXIF tags, camera info, generative traces)
- Explainable AI (XAI) summaries for model decisions
- Batch upload and analysis of multiple images at once
- Web interface built with Flask

---

## Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, Transformers, SigLIP, CNNs, ViTs
- **Frontend**: HTML, Jinja templates (minimal)
- **Visualization**: HTML reports, inference explanations
- **Storage**: Local filesystem (uploads + reports)

---

## Model Ensemble

- Base models: ResNet50, Vision Transformer, Custom CNN
- External model: [SigLIP](https://huggingface.co/Ateeqq/ai-vs-human-image-detector)
- Weighted ensemble for final prediction
- Ensemble weights can be trained using `train_ensemble_weights()`

---

## 🛠️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/sahannarachel/ai_image_detector.git
cd ai-image-detector
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

The app will start on `http://127.0.0.1:5000`

---

## Project Structure

```
├── app.py                     # Flask web server
├── ensemble.py                # Ensemble model logic
├── predict.py                 # Prediction and confidence logic
├── explainable_ai.py          # Model explanation generation
├── report.py                  # HTML report export functions
├── uploads/                   # Uploaded images
├── reports/                   # Generated HTML reports
├── templates/
│   ├── index.html             # Upload page
│   └── result.html            # (Optional) result display
├── requirements.txt
└── README.md
```

---

## How It Works

1. Upload image(s)
2. Image is processed through SigLIP and local models
3. Ensemble combines results into final prediction
4. Metadata and explanation are extracted
5. HTML report is generated with full analysis

---

## Batch Analysis Limits

- Default upload size: **16MB**
- You can increase this by editing:
  ```python
  app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64 MB
  ```

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- [SigLIP](https://huggingface.co/google/siglip-base-patch16-224)
- [Hugging Face Transformers](https://huggingface.co)
- [PyTorch](https://pytorch.org)
```
