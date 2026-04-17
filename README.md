# Aerial Rooftop Classification

A basic Computer Vision and Deep Learning pipeline for extracting, annotating, and classifying rooftop types (**Gable**, **Hip**, **Flat**) from large-scale aerial satellite imagery.

---

## 🚀 Project Overview

Classifying rooftop geometry from overhead satellite imagery is a non-trivial task — rooftops lack the directional gravity cues present in ground-level photographs, exhibit high inter-class similarity, and appear at varying scales across tiles. This project tackles that challenge through a systematic 3-phase pipeline:

1. **Computer Vision Extraction** — OpenCV connected-component analysis on binary building masks to isolate **2,294 individual rooftop crops** from 890 aerial tiles.
2. **Custom Annotation Toolkit** — A purpose-built Flask web application with keyboard shortcuts enabling rapid human labeling, producing a **100% human-verified ground truth** of **1,445 labeled crops**.
3. **Deep Learning Classification & Ablation Study** — A rigorous 6-part ablation study comparing 5 architectures, 4 augmentation strategies, 3 freeze configurations, 5 input resolutions, 4 dataset sizes, and weighted vs. unweighted sampling — totaling **23 distinct experiments**.

---

## 📂 Repository Structure

```
├── filtering_images.py                  # Filters tiles without rooftops
├── tif_to_jgp.py                        # Converts .tif masks to .jpg
├── extract_rooftop_crops.py             # Extracts individual rooftop crops
├── labeling_tool.py                     # Flask-based annotation web app
├── train_classifier.py                  # ResNet18 fine-tuning pipeline
├── test_classifier.py                   # Evaluation on verified dataset
├── Rooftop_Classification_Study_Final.ipynb  # Master ablation notebook
├── requirements.txt                     # Python dependencies
├── Rooftop_Crops/                       # Cropped images + labels.json
├── model/                               # Trained model weights (.pth)
└── study_results/                       # 9 ablation study plots + metrics
```

### Data Processing Pipeline
| File | Purpose |
|------|---------|
| `filtering_images.py` | Scans raw satellite masks and filters out tiles with no rooftop content |
| `tif_to_jgp.py` | Converts geospatial `.tif` mask files to standard `.jpg` format |
| `extract_rooftop_crops.py` | Uses connected-component analysis to extract 2,294 padded rooftop crops |

### Annotation Toolkit
| File | Purpose |
|------|---------|
| `labeling_tool.py` | Flask web app with keyboard-driven annotation (1=Gable, 2=Hip, 3=Flat) |

### Classification & Evaluation
| File | Purpose |
|------|---------|
| `train_classifier.py` | Fine-tunes ResNet18 with augmentation + weighted sampling |
| `test_classifier.py` | Evaluates trained model on the full human-verified dataset |
| `Rooftop_Classification_Study_Final.ipynb` | Complete 6-part ablation study notebook (run on Colab with GPU) |

---

## 📊 Key Results

### Architecture Comparison

![Architecture Comparison](study_results/1_architecture_comparison.png)

| Architecture | Accuracy | F1 Score | Parameters |
|---|---|---|---|
| ResNet-50 | **83.1%** | **0.833** | 22.07M |
| MobileNetV3-Small | 81.5% | 0.819 | 1.27M |
| ResNet-18 | 80.4% | 0.806 | 10.49M |
| EfficientNet-B0 | 79.4% | 0.799 | 3.16M |
| Simple CNN (baseline) | 57.7% | 0.522 | 0.39M |

### Dataset Statistics
| Class | Count | Proportion |
|---|---|---|
| Hip | 697 | 48.2% |
| Gable | 509 | 35.2% |
| Flat | 239 | 16.5% |
| **Total** | **1,445** | 100% |

### Ablation Study Highlights
- **Weighted sampling** improved accuracy from 79.4% → 83.1% (+3.7%)
- **160×160 input resolution** outperformed both smaller (64px: 69.3%) and larger (224px: 81.5%) sizes
- **Heavy augmentation** (rotation-invariant transforms) yielded the best generalization for aerial imagery
- **Partial freezing** matched full fine-tuning at half the compute cost
- **Full freezing** (linear probe only) collapsed to 55.6%, confirming ImageNet features alone are insufficient for overhead geometry

---

## 🛠️ Usage

### Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

### Running the Ablation Study (Google Colab)
1. Upload `Rooftop_Classification_Study_Final.ipynb` and `Rooftop_Crops/` (as a zip) to Colab.
2. Set runtime to **T4 GPU**.
3. Add a cell at the top: `!unzip -q Rooftop_Crops.zip`
4. Run all cells.

### Running the Labeling Tool
```bash
python labeling_tool.py
# Open http://localhost:5555 in your browser
```
