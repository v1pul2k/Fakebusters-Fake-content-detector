# 🛡️ FAKEBUSTERS

### Multi-Modal Deepfake Detection System (Image • Video • News)

---

## 🚀 Overview

**FAKEBUSTERS** is an end-to-end AI system designed to detect **deepfake content across multiple modalities** — images, videos, and textual news.
It leverages **deep learning + machine learning ensembles** to deliver high-accuracy, real-time detection for combating misinformation.

---

## 🎯 Key Highlights

* ⚡ Built a **multi-modal detection pipeline** (Image + Video + Text)
* 🧠 Achieved **~92–95% accuracy (images)** and **~90% (videos)**
* 🎥 Designed **temporal video analysis** using frame sampling + feature aggregation
* 📊 Implemented **ensemble ML models** for fake news detection (~92% accuracy)
* 🌐 Developed a **real-time Streamlit web app** for user interaction
* 🎯 Applied **threshold optimization (Youden’s Index)** for balanced predictions

---

## 🧠 Technical Stack

* **Languages:** Python
* **Deep Learning:** PyTorch, torchvision
* **ML Models:** Random Forest, Gradient Boosting, Logistic Regression
* **Computer Vision:** OpenCV
* **Deployment:** Streamlit
* **Data Processing:** NumPy, scikit-learn

---

## 🏗️ Architecture

```text
Input → Preprocessing → Feature Extraction → Model Inference → Prediction + Confidence
```

### 🔹 Image Pipeline

* ResNet-18 (Transfer Learning)
* Data Augmentation + Normalization

### 🔹 Video Pipeline

* Xception Backbone
* Frame Sampling (8 frames/video)
* Temporal Averaging (efficient alternative to RNN)

### 🔹 Text Pipeline

* Feature Engineering
* Ensemble Learning (RF + GB + LR + DT)

---

## 📊 Performance Metrics

| Module          | Accuracy | F1 Score | ROC-AUC |
| --------------- | -------- | -------- | ------- |
| Image Detection | 92–95%   | ~0.92    | ~0.96   |
| Video Detection | 88–91%   | ~0.88    | ~0.93   |
| Fake News       | ~92%     | ~0.91    | ~0.94   |

---

## 💡 What Makes This Project Strong

* ✅ **Multi-modal approach** (rare + high-impact)
* ✅ **Production-style pipeline** (not just model training)
* ✅ **Real-time deployment (Streamlit UI)**
* ✅ **Performance tuning + threshold optimization**
* ✅ **Balanced metrics (precision, recall, ROC-AUC)**

---

## ⚙️ How to Run

```bash
git clone https://github.com/your-username/fakebusters.git
cd fakebusters
pip install -r requirements.txt
streamlit run app.py
```

---

```

---

## 🔥 Impact

* Helps **detect AI-generated misinformation**
* Useful for:

  * Social media moderation
  * News verification
  * Digital forensics

---

## ⚠️ Limitations

* Generalization depends on dataset (Celeb-DF)
* High-quality deepfakes may bypass detection
* Requires GPU for optimal performance

---

## 🔮 Future Enhancements

* Vision Transformers (ViT)
* Audio-visual deepfake detection
* Explainable AI (Grad-CAM)
* API / browser extension deployment

---

## 👨‍💻 Contributors

* Samridh Sagar
* Vipul Singh
* Adyatan Agarwal



---
