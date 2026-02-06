# 🏠 HomeSafe  
**Intelligent Face Recognition & Fall Detection System**

HomeSafe is an intelligent computer vision system that combines **facial recognition** and **fall detection** in real time.  
The project demonstrates how **Convolutional Neural Networks (CNNs)** and **deep learning pipelines** can be applied to real-world safety and surveillance scenarios.

This project was developed as part of the **Convolutional Neural Networks (CNN)** course.

---

## 🎯 Project Objectives

HomeSafe aims to:

- 👤 Recognize authorized household members using facial recognition  
- 🚨 Detect intrusions by identifying unknown faces  
- 🧍 Detect human falls in real time using body posture analysis  
- 🖥️ Provide a real-time interactive interface via **Streamlit**

---

## 🧠 Technologies Used

### 🔹 Computer Vision
- **MTCNN** – Face detection  
- **MediaPipe Pose** – Human pose extraction (33 body landmarks)

### 🔹 Deep Learning & Machine Learning
- **FaceNet (CNN, pre-trained)** – Facial embeddings extraction  
- **SVM** – Face identity classification  
- **CNN 1D** – Fall detection based on pose landmarks  

### 🔹 Tools & Frameworks
- Python
- OpenCV
- TensorFlow / Keras
- PyTorch
- Streamlit
- NumPy, Scikit-learn

---

## 🏗️ System Architecture

### 🔹 Facial Recognition Pipeline
Camera → MTCNN → FaceNet → SVM → Identity Decision
### 🔹 Fall Detection Pipeline
Camera → MediaPipe Pose → CNN 1D → Fall / Normal Decision
The architecture is **modular**, allowing:
- Independent testing of each component  
- Easier maintenance  
- Progressive integration  

---

## 📊 Dataset & Training

### 🔹 Fall Detection Dataset
- Dataset: **Le2i Fall Detection Dataset (Kaggle)**
- Original format: videos
- Processing steps:
  - Video-to-frame extraction
  - Manual selection of fall intervals using annotation files
  - Classification into `normal` and `fall`

Each frame is converted into **99 numerical features**  
(33 landmarks × x, y, z coordinates).

### 🔹 Model Training
- CNN 1D trained from scratch for fall detection
- Binary classification: `normal / fall`
- Train/Test split
- Early stopping used to prevent overfitting

### 🔹 Facial Recognition
- FaceNet is **pre-trained**
- SVM classifier is trained dynamically using registered faces

> ⚠️ **Note:**  
Datasets, frames, videos, trained models, and personal images are intentionally excluded from this repository for **privacy and size reasons**.

---

## 💻 Implementation

The system was implemented step by step:

1. Individual module testing via terminal:
   - Frame extraction
   - Pose extraction
   - Embedding generation
   - Model training
2. Progressive integration
3. Final deployment using **Streamlit**

Performance optimizations include:
- Processing 1 frame out of 3
- Confidence thresholds tuning
- Error handling (`try/except`)

---

## 🖥️ Application Interface

The Streamlit interface allows:
- Registering new members
- Real-time face recognition
- Real-time fall detection
- Event logging

---

## 📂 Project Structure
HomeSafe/
│── face_recognition/
│── fall_detection/
│── alerts/
│── models/
│── embeddings/
│── streamlit_app.py
│── config.py
│── requirements.txt
│── README.md
> ⚠️ Some folders are ignored via `.gitignore` (datasets, models, logs).

---

## 🚀 Installation & Usage

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the application
```bash
streamlit run streamlit_app.py
```

## 🎓 Key Learnings
- Choosing the right CNN architecture depends on data structure
-	Preprocessing is as important as the model itself
-	Transfer learning significantly speeds up development
-	Modular architecture simplifies testing and scalability
