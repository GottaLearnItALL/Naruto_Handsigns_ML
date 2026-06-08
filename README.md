# 🍥 Naruto Hand Sign Detection with Visual Jutsu Effects

Real-time computer vision system that detects Naruto hand signs using webcam input and triggers animated visual jutsu effects. Built from scratch with classical ML models — no deep learning black boxes.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-red)

## Demo

> *Coming soon — screen recording of live detection with fireball, chidori, and shadow clone effects*

## How It Works

The system follows a four-stage pipeline:

**1. Hand Detection** — MediaPipe Hands detects 21 hand landmarks in real-time from the webcam feed, providing (x, y, z) coordinates for each point on the hand.

**2. Feature Extraction** — The 21 landmarks are flattened into a 63-dimensional feature vector (21 × 3 coordinates). Each vector is normalized by subtracting the wrist position, making the features translation-invariant — the model learns hand *shape*, not hand *position* on screen.

**3. Classification** — An SVM classifier with RBF kernel predicts which of 13 Naruto hand signs is being performed. The model was selected after comparing KNN (built from scratch), SVM, and Random Forest.

**4. Jutsu Activation** — A state machine tracks the sequence of detected signs. When a known jutsu combo is completed (e.g., Snake → Ram → Dog for Fireball), the corresponding visual effect is triggered and rendered on the webcam feed.

## Model Comparison

Three classifiers were trained and evaluated on the same 80/20 train-test split:

| Model | Accuracy | Notes |
|-------|----------|-------|
| KNN (from scratch, NumPy) | 96.0% | K=5, Euclidean distance, no libraries |
| Random Forest | 96.6% | 100 estimators, sklearn |
| **SVM (RBF, C=100)** | **96.9%** | **Selected model** — best via GridSearchCV |

Hyperparameter tuning was performed using 5-fold cross-validation with GridSearchCV, testing C ∈ {0.1, 1, 10, 100} across RBF and linear kernels.

## Jutsu Effects

| Jutsu | Combo | Effect |
|-------|-------|--------|
| 🔥 Fireball (Katon) | Snake → Ram → Dog | Expanding fireball with particle embers, hand-tracked |
| ⚡ Chidori | Ox → Hare → Monkey | Lightning bolts with electric sparks, hand-tracked |
| 👥 Shadow Clone | Ram → Snake | Screen splits into multiple copies using selfie segmentation |

Effects are rendered in real-time using OpenCV drawing primitives with alpha blending. A stability filter requires the same sign to be held for 10 consecutive frames before registration, preventing jitter during hand transitions.

## Tech Stack

- **MediaPipe Hands** — 21-landmark hand pose estimation
- **MediaPipe Selfie Segmentation** — person isolation for shadow clone effect
- **OpenCV** — webcam capture, image processing, visual effects rendering
- **scikit-learn** — SVM, Random Forest, GridSearchCV, train/test split
- **NumPy** — custom KNN implementation from scratch, feature processing
- **Streamlit** — web demo interface
- **joblib** — model serialization

## Project Structure

```
naruto-signs/
├── data/
│   ├── train/                  # 13 sign classes, ~2000 images
│   ├── test/                   # Held-out test set
│   └── features_train.npz     # Extracted 63-dim landmark vectors
├── models/
│   └── svm_model.pkl           # Trained SVM classifier
├── src/
│   ├── knn_scratch.py          # KNN from scratch (NumPy only)
│   ├── extract_features.py     # MediaPipe landmark extraction pipeline
│   ├── train_models.py         # Model training + comparison
│   ├── live_predict.py         # Real-time webcam prediction + effects
│   └── collect_data.py         # Webcam data collection utility
├── app.py                      # Streamlit demo app
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/Gottalearnitall/naruto-hand-sign-detection.git
cd naruto-hand-sign-detection

# Create environment (requires Python 3.12)
uv venv --python 3.12
source .venv/bin/activate  # or .venv\Scripts\Activate.ps1 on Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

**Run the live detector:**
```bash
python src/live_predict.py
```
Perform hand signs in sequence to activate jutsus. Press `q` to quit.

**Train models from scratch:**
```bash
python src/extract_features.py    # Extract landmarks from dataset
python src/train_models.py        # Train and compare KNN, SVM, RF
```

**Run the Streamlit demo:**
```bash
streamlit run app.py
```

## Dataset

[Naruto Hand Sign Dataset](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset) from Kaggle — 13 classes (Bird, Boar, Dog, Dragon, Hare, Horse, Monkey, Ox, Ram, Rat, Snake, Tiger, Zero) with ~2000 total images. After MediaPipe landmark extraction, 1765 samples successfully produced 63-dimensional feature vectors.

## Key ML Concepts Applied

- **Feature engineering** — raw images (150K+ dimensions) reduced to 63 meaningful landmark features via MediaPipe, enabling classical ML models to achieve 96%+ accuracy
- **Translation invariance** — wrist-relative normalization ensures hand shape is captured regardless of position in frame
- **Hyperparameter tuning** — GridSearchCV with 5-fold cross-validation for SVM kernel and C parameter selection
- **Curse of dimensionality** — feature extraction from high-dimensional image space to low-dimensional landmark space
- **Model selection** — systematic comparison of KNN, SVM, and Random Forest on identical train/test splits
- **Bias-variance tradeoff** — SVM C parameter controls margin width vs classification strictness

## Future Improvements

- CNN-based classifier for direct image input (bypass MediaPipe)
- LSTM/sequence model for temporal sign recognition
- Additional jutsu combos and effects
- Confidence thresholding for out-of-distribution gesture rejection
- Web deployment with streamlit-webrtc for real-time browser demo

## License

MIT