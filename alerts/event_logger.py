import csv
from datetime import datetime
import os
import config

LOG_FILE = config.EVENT_LOG_PATH

LOG_FILE = "alerts/events_log.csv"

def log_event(event_type, description):
    """
    event_type : FALL / INTRUS / FACE / SYSTEM
    description : message lisible
    """
    os.makedirs("alerts", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)

        # écrire l’en-tête une seule fois
        if not file_exists:
            writer.writerow(["timestamp", "event_type", "description"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            event_type,
            description
        ])