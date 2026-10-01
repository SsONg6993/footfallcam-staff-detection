from ultralytics import YOLO

MODEL_PATH = (
    "runs/detect/runs/badge/"
    "yolo26n_badge-4/weights/best.pt"
)

VAL_IMAGES = (
    "outputs/badge_dataset_v2/images/val"
)

model = YOLO(MODEL_PATH)

model.predict(
    source=VAL_IMAGES,
    imgsz=416,
    conf=0.25,
    save=True,
    project="runs/badge_test",
    name="validation_predictions"
)

print("Done.")
print(
    "Check: "
    "runs/badge_test/validation_predictions/"
)