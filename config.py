# =========================
# HomeSafe - Configuration
# =========================

# ---------
# Sécurité Flask
# ---------
SECRET_KEY = "homesafe_secret_key"

# ---------
# Caméra
# ---------
CAMERA_INDEX = 0           # 0 = webcam par défaut
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ---------
# Reconnaissance faciale
# ---------
FACE_CONFIDENCE_THRESHOLD = 0.5   # seuil pour UNKNOWN / membre reconnu
FACE_DATASET_PATH = "dataset"
FACE_MODEL_PATH = "models/face_classifier.pkl"

# ---------
# Détection de chute
# ---------
FALL_DURATION_ADULT = 2.0   # secondes au sol avant alerte
FALL_DURATION_BABY = 1.5

HIP_Y_THRESHOLD = 0.75      # seuil posture au sol
SHOULDER_Y_THRESHOLD = 0.65

FALL_MODEL_PATH = "models/fall_model.h5"

# ---------
# Historique des événements
# ---------
EVENT_LOG_PATH = "alerts/events_log.csv"

# ---------
# Interface
# ---------
APP_NAME = "HomeSafe"
APP_DESCRIPTION = "Système intelligent de surveillance et d’assistance domestique"

# ---------
# Mode debug
# ---------
DEBUG_MODE = True