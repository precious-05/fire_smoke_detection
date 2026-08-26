# Fire and Smoke Detection

<div align="center">

<p align="center">
  <!-- Video demo retained from original README -->
  <strong>Video demo</strong>
</p>

<p align="center">
  <a href="https://github.com/user-attachments/assets/e6bf4054-79ab-4923-b0b2-db92b802a909" target="_blank">Watch the demo video</a>
</p>

<p align="center">
  <em>Real-time object detection model to detect Fire and Smoke from images and video</em>
</p>

<p align="center">
  <!-- Tech stack icons (real icons) -->
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="Python" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/jupyter/jupyter-original-wordmark.svg" alt="Jupyter" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" alt="PyTorch" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" alt="OpenCV" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="NumPy" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" alt="Matplotlib" width="36" />
  <img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pandas/pandas-original.svg" alt="Pandas" width="36" />
</p>

</div>

---

## Overview

This repository contains `fire_smoke_detection.ipynb`, a Jupyter notebook that implements an end-to-end object detection workflow for detecting Fire and Smoke. The notebook includes dataset validation, model configuration, training, evaluation, plotting, and inference examples.

Classes: Fire, Smoke

---

## Repository layout (expected)

- fire_smoke_detection.ipynb  - main notebook with full pipeline
- data/
  - train/images
  - train/labels
  - valid/images
  - valid/labels
  - test/images
  - test/labels
- models/                     - trained weights (for example: best.pt)
- results/                    - training plots, CSV metrics

Adjust paths in the notebook if your layout differs.

---

## Requirements

The notebook was run on Python 3.12 kernel but works with Python 3.8 and above. Install required libraries in a virtual environment. Example:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip
pip install jupyterlab numpy pandas matplotlib seaborn opencv-python pillow torch torchvision
# If using ultralytics YOLO implementation
pip install ultralytics
```

---

## Quick start

1. Prepare dataset
   - Use YOLO txt format or COCO. Notebook assumes YOLO-style labels: `class x_center y_center width height` (normalized).
   - Example structure: `data/train/images`, `data/train/labels`, `data/valid/images`, `data/valid/labels`.

2. Open the notebook:

```bash
jupyter lab
# or
jupyter notebook
```

3. Edit configuration cells at the top of the notebook to set:
   - dataset paths
   - classes
   - model backbone or pretrained weights
   - hyperparameters: epochs, imgsz, batch, lr

4. Run cells in order to train, evaluate, and run inference.

---

## What is included in the notebook

- Data validation and visualization utilities
- Model loading and training flow
- Training metrics plots and evaluation (precision, recall, F1, PR curves)
- Single image and video inference with OpenCV drawing helpers
- Examples of saving model weights and exporting results

---

## Example usage snippets

Train (example pseudocode adapted from the notebook):

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=20, imgsz=640, batch=16, device=0)
```

Predict single image:

```python
results = model.predict(source='path/to/image.jpg', conf=0.25)
# use the notebook helper to draw boxes and labels
```

Evaluate on test set:

```python
metrics = model.val(data='data.yaml', split='test')
```

---

## Tips and troubleshooting

- Confirm label format matches the model API expectations.
- For GPU memory issues, reduce `imgsz` or `batch` and consider gradient accumulation.
- To speed up experiments use smaller backbone models like `yolov8n`.
- The notebook contains checks that visualize samples and verify label consistency. Run those before training.

---

## Next steps I can perform

- Add `requirements.txt` or `environment.yml` for reproducible setup
- Commit training artifacts or sample results to `models/` or `results/`
- Extract notebook steps into modular Python scripts for CLI usage

If you want me to commit any of the above or change the demo link text, tell me which action to take and I will apply it.
