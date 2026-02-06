import streamlit as st
import cv2
import numpy as np
import pickle
import joblib#  chargement des modèles SVM
import os
from face_recognition.detect_face import FaceDetector
from face_recognition.embedder import FaceEmbedder
from alerts.event_logger import log_event
import config
from face_recognition.train_classifier import train_classifier
import mediapipe as mp
import time
from tensorflow.keras.models import load_model

# =========================
# CONFIG STREAMLIT
# =========================

st.set_page_config(
    page_title="HomeSafe",
    layout="wide"
)
st.title("🏠 HomeSafe – Reconnaissance Faciale")
st.caption("Système intelligent de reconnaissance faciale et détection d’intrusion basé sur CNN")

# =========================
# LOAD MODELS
# =========================

@st.cache_resource
def load_models():

    clf = joblib.load("models/classifier.pkl")
    le = joblib.load("models/label_encoder.pkl")
    return clf, le


classifier, label_encoder = load_models()

# =========================
# LOAD FALL MODEL
# =========================
@st.cache_resource
def load_fall_model():
    return load_model("models/fall_model.h5")

fall_model = load_fall_model()

# 🔹 INITIALISATION DU DÉTECTEUR DE VISAGES
face_detector = FaceDetector()
face_embedder = FaceEmbedder()

# =========================
# INITIALISATION MEDIAPIPE POSE
# =========================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# FEATURE EXTRACTION FUNCTION
# =========================
def extract_pose_features(frame):
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    features = []
    for lm in results.pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features)

# =========================
# SIDEBAR
# =========================

mode = st.sidebar.radio(
    "Choisissez un mode",
    ["Reconnaissance faciale", 
     "➕ Enregistrer un membre", 
     "🚨 Détection de chute"]
)

if mode == "➕ Enregistrer un membre":
    st.subheader("➕ Enregistrement d’un nouveau membre")

    prenom = st.text_input("Prénom")
    nom = st.text_input("Nom")

    start_cam = st.checkbox("📷 Activer la caméra")

    preview = st.empty()

    if start_cam:
        cap = cv2.VideoCapture(config.CAMERA_INDEX)

        # 🔹 laisser le temps à la caméra de s'initialiser
        time.sleep(1.0)
        
        # Capturer plusieurs frames pour laisser la caméra s'ajuster
        frame = None
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                time.sleep(0.1)
        
        cap.release()

        if ret and frame is not None and frame.size != 0:
            preview.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Aperçu caméra",
                use_container_width=True
            )
        else:
            st.error("Impossible d'accéder à la caméra (image vide)")

    if st.button("📸 Capturer et enregistrer"):
        if prenom == "" or nom == "":
            st.error("Veuillez remplir tous les champs.")
        elif not start_cam:
            st.error("Veuillez activer la caméra.")
        else:
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                st.error("Erreur caméra")
            else:
                boxes, probs = face_detector.detect_faces(frame)

                if len(boxes) == 0:
                    st.error("Aucun visage détecté")
                else:
                    x1, y1, x2, y2 = map(int, boxes[0])
                    face = frame[y1:y2, x1:x2]

                    member_name = f"{prenom}_{nom}"
                    save_dir = f"dataset/{member_name}"
                    os.makedirs(save_dir, exist_ok=True)

                    cap = cv2.VideoCapture(config.CAMERA_INDEX)

                    saved = 0
                    MAX_IMAGES = 7   # ✅ nombre d’images par personne

                    while saved < MAX_IMAGES:
                        ret, frame = cap.read()
                        if not ret:
                            continue

                        boxes, probs = face_detector.detect_faces(frame)
                        if len(boxes) == 0:
                           continue

                        x1, y1, x2, y2 = map(int, boxes[0])
                        face = frame[y1:y2, x1:x2]

                        if face.size == 0:
                            continue

                        img_path = f"{save_dir}/{member_name}_{saved}.jpg"
                        cv2.imwrite(img_path, face)

                        saved += 1
                        time.sleep(0.4)  # petite pause entre les images

                    cap.release()

                    st.success(f"✅ Visage enregistré pour {member_name}")
                    st.image(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    st.info("🔄 Mise à jour du modèle...")
                    face_embedder.process_dataset()
                    train_classifier()

                    st.success("🎉 Membre enregistré et reconnaissable")

# =========================
# CAMERA
# =========================
run = False
if mode == "Reconnaissance faciale":
    run = st.checkbox("▶️ Démarrer la caméra")
FRAME_WINDOW = st.image([])
cap = None
if mode == "Reconnaissance faciale":
    cap = cv2.VideoCapture(config.CAMERA_INDEX)


if run and cap is not None:
    frame_count = 0
    last_boxes = []
    last_probs = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Erreur caméra")
            break

        frame_count += 1
        
        # 🚀 Optimisation: traiter 1 frame sur 3 pour la détection
        if frame_count % 3 == 0:
            boxes, probs = face_detector.detect_faces(frame)
            last_boxes = boxes
            last_probs = probs
        else:
            boxes = last_boxes
            probs = last_probs

        for (box, prob) in zip(boxes, probs):
            x1, y1, x2, y2 = map(int, box)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            embedding = face_embedder.get_embedding_from_frame(face)

            # Si aucun embedding n'est retourné (visage mal détecté)
            if embedding is None:
                continue

            embedding = embedding.reshape(1, -1)

            probs_pred = classifier.predict_proba(embedding)[0]
            max_prob = np.max(probs_pred)
            label_idx = np.argmax(probs_pred)
            name = label_encoder.inverse_transform([label_idx])[0]

            if max_prob < config.FACE_CONFIDENCE_THRESHOLD:
                label = "UNKNOWN"
                color = (0, 0, 255)
                log_event("INTRUS", "Visage inconnu détecté")
            else:
                label = name
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{label} ({max_prob:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    if cap is not None:
        cap.release()

# ======================================================
# 🚨 MODE : DÉTECTION DE CHUTE
# ======================================================

if mode == "🚨 Détection de chute":

    st.subheader("🚨 Détection de chute en temps réel")

    run_fall = st.checkbox("▶️ Démarrer la détection")

    frame_box = st.image([])

    cap = cv2.VideoCapture(config.CAMERA_INDEX)

    fall_start_time = None
    FALL_DURATION_THRESHOLD = 2.0

    if run_fall:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Erreur caméra")
                break

            features = extract_pose_features(frame)

            status = "🟢 NORMAL"
            color = (0, 255, 0)

            if features is not None:
                X = features.reshape(1, -1, 1)
                score = fall_model.predict(X, verbose=0)[0][0]

                if score > 0.6:
                    if fall_start_time is None:
                        fall_start_time = time.time()

                    elapsed = time.time() - fall_start_time

                    if elapsed >= FALL_DURATION_THRESHOLD:
                        status = "🔴 CHUTE DÉTECTÉE"
                        color = (0, 0, 255)
                        log_event("FALL", "Chute détectée")
                    else:
                        status = "🟠 CHUTE POSSIBLE"
                        color = (0, 165, 255)
                else:
                    fall_start_time = None

            cv2.putText(
                frame,
                status,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                color,
                3
            )

            frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    else:
        cap.release()