from pathlib import Path


# Input / output
VIDEO_PATH = "sample.mp4"

OUTPUT_DIR = Path("outputs/final")

OUTPUT_VIDEO = OUTPUT_DIR / "staff_detection.mp4"
OUTPUT_CSV = OUTPUT_DIR / "staff_detection.csv"


# Person detector
PERSON_MODEL = "models/person_best.pt"

PERSON_CONF = 0.25
PERSON_IMGSZ = 768


# Badge detector
BADGE_MODEL = "models/badge_best.pt"

BADGE_CONF = 0.5
BADGE_IMGSZ = 416
BADGE_CHECK_INTERVAL = 1

# Tracking
MAX_MISSED_FRAMES = 20
TRACK_MAX_CENTER_DISTANCE = 1.5
TRACK_MIN_IOU = 0.05


# Static false-positive check
STATIC_CHECK_MIN_AGE = 75
STATIC_MAX_SPREAD = 8.0
STATIC_MAX_AVG_CONF = 0.35
