from ultralytics import YOLO


DATA_YAML = "outputs/badge_dataset_v2/data.yaml"

MODEL_NAME = "yolo26n.pt"

EPOCHS = 80
IMAGE_SIZE = 416
BATCH_SIZE = 8
PATIENCE = 20


def main():
    print("Loading model...")

    model = YOLO(MODEL_NAME)

    print("Model loaded.")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        device=0,
        workers=4,
        project="runs/badge",
        name="yolo26n_badge"
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Best model should be at:"
    )

    print(
        "runs/badge/"
        "yolo26n_badge/"
        "weights/best.pt"
    )


if __name__ == "__main__":
    main()