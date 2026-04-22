# Skin Disease Mini Project

This repository has been cleaned up and updated into a simple 2025-ready mini project for HAM10000 skin disease classification.

The project now focuses on three things only:

1. Basic dataset handling for HAM10000
2. A lightweight PyTorch model pipeline
3. A simple Streamlit interface for image upload and prediction

It is designed for:

- Python 3.10+
- CPU-only systems
- Easy setup for college demos
- Clean code that is easier to explain in a viva or presentation

## Updated Folder Structure

```text
Skin_disease/
|-- app.py
|-- predict.py
|-- train.py
|-- requirements.txt
|-- README.md
|-- configs/
|   `-- default.json
|-- model/
|   `-- model_resnet18.pth
`-- src/
    `-- skin_disease/
        |-- __init__.py
        |-- config.py
        |-- constants.py
        |-- data.py
        |-- inference.py
        |-- model.py
        |-- training.py
        `-- utils.py
```

## What Was Removed and Why

- Flask API server: unnecessary for a mini project when Streamlit already gives a clean demo UI
- Docker and docker-compose: adds setup complexity without helping a college-level demonstration
- LIME explanation pipeline: too slow and heavy for CPU-only demo systems
- Duplicate English/French frontends: one simple UI is enough
- Old CLI client/server request flow: replaced with direct local inference
- Notebook-based training flow: replaced with a cleaner script-based pipeline
- Extra ResNet34 checkpoint: removed to keep the project smaller and more CPU-friendly

## What Was Updated and Why

- Python target upgraded to 3.10+
- Dependencies refreshed into a single `requirements.txt`
- Code reorganized into a reusable package under `src/skin_disease`
- Dataset loading rewritten to support common HAM10000 folder layouts
- Inference pipeline simplified to direct local model loading
- Training script now saves a cleaner checkpoint format with metadata
- Class mapping fixed for melanoma

## Important Note About the Existing Checkpoint

The project still uses the existing `model/model_resnet18.pth` checkpoint so the demo can run quickly.

However, this is a legacy checkpoint from the older project. The new code can load it, but for the cleanest result you should retrain once using `train.py` and your HAM10000 dataset folder.

## Dataset Layout

The training script expects a folder containing:

- `HAM10000_metadata.csv`
- image folders such as `HAM10000_images_part_1` and `HAM10000_images_part_2`

It also works with older train/test image folders if they already exist.

Example:

```text
HAM10000/
|-- HAM10000_metadata.csv
|-- HAM10000_images_part_1/
`-- HAM10000_images_part_2/
```

## Setup Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, upload an image, and click `Predict`.

### 4. Optional: run prediction from the command line

```bash
python predict.py path/to/image.jpg
```

### 5. Optional: retrain the model

```bash
python train.py --data-dir path/to/HAM10000
```

The best checkpoint will be saved to:

```text
model/model_resnet18.pth
```

## Simple Demo Flow

For a mini project presentation, the easiest flow is:

1. Open the Streamlit app
2. Upload a skin image
3. Show the predicted disease name
4. Show the confidence score
5. Explain that the model is trained on HAM10000 using transfer learning

## Tech Stack

- PyTorch
- Torchvision
- Streamlit
- Pandas
- Scikit-learn
- Pillow

## Mini Project Limitations

- This is a student demo, not a medical diagnosis tool
- Prediction quality depends on the training data and checkpoint quality
- CPU inference is supported, but training on CPU can still take time

## Suggested Future Improvements

- Add class-wise performance charts
- Save prediction history
- Add Grad-CAM visual explanations
- Export results to PDF for project reports
