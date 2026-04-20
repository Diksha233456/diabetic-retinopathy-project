# 👁️ Diabetic Retinopathy Detection

A clean, modular **inference pipeline** for detecting diabetic retinopathy severity from retinal fundus images. Built with PyTorch + Streamlit.

> ⚠️ **This is a project skeleton.** It ships with a dummy model for pipeline validation. Replace the model with a trained backbone before clinical use.

---

## 🗂️ Project Structure

```
dr-retinopathy-project/
├── app/
│   └── app.py               # Streamlit web app
├── model/
│   ├── model_loader.py      # DummyDRModel definition + load_model()
│   └── predict.py           # Inference pipeline → label + probabilities
├── utils/
│   ├── preprocess.py        # Image → (1, 3, 224, 224) tensor
│   └── gradcam.py           # Grad-CAM placeholder (future)
├── data/
│   └── sample_images/       # Drop test images here
├── saved_models/
│   └── model.pth            # Trained weights go here (see saved_models/README.txt)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/dr-retinopathy-project.git
cd dr-retinopathy-project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Generate dummy model weights

```bash
python -c "
import torch
from model.model_loader import DummyDRModel
model = DummyDRModel()
torch.save(model.state_dict(), 'saved_models/model.pth')
print('Saved dummy model.pth')
"
```

---

## 🚀 Run the App

```bash
streamlit run app/app.py
```

Then open **http://localhost:8501** in your browser, upload any JPG/PNG image, and see the prediction.

---

## 🔬 DR Severity Classes

| Grade | Label             | Description                              |
|-------|-------------------|------------------------------------------|
| 0     | No DR             | No signs of diabetic retinopathy         |
| 1     | Mild              | Microaneurysms only                      |
| 2     | Moderate          | More than mild, less than severe         |
| 3     | Severe            | Extensive haemorrhages, no PDR           |
| 4     | Proliferative DR  | Neovascularisation — most severe stage   |

---

## 🧩 Module Overview

| File                    | Responsibility                                              |
|-------------------------|-------------------------------------------------------------|
| `utils/preprocess.py`   | OpenCV image load → resize → normalise → CHW batch tensor  |
| `model/model_loader.py` | DummyDRModel (FC) + `load_model()` with optional .pth load |
| `model/predict.py`      | Tensor inference → softmax → top-1 label + prob dict       |
| `app/app.py`            | Streamlit UI: upload → preprocess → predict → display      |
| `utils/gradcam.py`      | Grad-CAM placeholder (implement in next milestone)          |

---

## 🛣️ Roadmap

- [ ] Replace `DummyDRModel` with EfficientNet-B4 backbone
- [ ] Add training pipeline (`train/`)
- [ ] Implement Grad-CAM heatmap overlay
- [ ] Add model evaluation metrics (AUC, Cohen's Kappa)
- [ ] Dockerise the app

---

## 📄 License

MIT — free to use and modify for research and educational purposes.
