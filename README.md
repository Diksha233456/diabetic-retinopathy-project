# 👁️ Diabetic Retinopathy Detection (Deep Learning Project)

This project is a modular deep learning system designed to classify retinal fundus images into different stages of **Diabetic Retinopathy (DR)**.

It includes a complete **inference pipeline + web application**, with a structure ready for integration of a trained CNN model.

---

## 🚀 Features

* Upload retinal images via Streamlit UI
* Modular inference pipeline (preprocess → model → prediction)
* Class-wise probability outputs
* Confidence score display
* Clean and scalable project structure
* Ready for real model integration

---

## 🧠 DR Classification Classes

The system predicts one of the following:

* No Diabetic Retinopathy
* Mild Diabetic Retinopathy
* Moderate Diabetic Retinopathy
* Severe Diabetic Retinopathy
* Proliferative Diabetic Retinopathy

---

## 📁 Project Structure

dr-retinopathy-project/

├── app/                # Streamlit UI
│   └── app.py

├── model/              # Model loading & prediction
│   ├── model_loader.py
│   └── predict.py

├── utils/              # Preprocessing + utilities
│   ├── preprocess.py
│   └── gradcam.py

├── saved_models/       # Model weights
│   └── model.pth

├── data/               # Sample images (optional)

├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation & Setup

git clone https://github.com/YOUR_USERNAME/diabetic-retinopathy-project.git
cd diabetic-retinopathy-project

pip install -r requirements.txt

---

## ▶️ Run the Application

python -m streamlit run app/app.py

---

## ⚠️ Current Status

* The system currently uses a **dummy model (placeholder)**
* Predictions are **not medically accurate yet**
* Designed to validate pipeline and UI

---

## 🧪 Generate Dummy Model Weights (Optional)

python -c "import torch; from model.model_loader import DummyDRModel; model = DummyDRModel(); torch.save(model.state_dict(), 'saved_models/model.pth')"

---

## 🔬 Future Work

* Integrate trained CNN (ResNet / EfficientNet)
* Add Grad-CAM for explainability
* Improve UI for clinical usability
* Add evaluation metrics (Accuracy, F1-score, AUC)

---

## ⚠️ Model Files

* Do NOT commit large `.pth` files to GitHub
* Use Git LFS or DVC for model versioning

---

## 👨‍💻 Team Roles

* Diksha → UI + Inference Pipeline + Integration
* Likith → Model Training (CNN)
* Inchara → Data + Preprocessing
* Vigneshwara → Evaluation + Metrics

---

## ⚠️ Disclaimer

This project is for **educational purposes only** and should NOT be used for real medical diagnosis.

---

## ⭐ Contribution

Feel free to fork and improve the project!
