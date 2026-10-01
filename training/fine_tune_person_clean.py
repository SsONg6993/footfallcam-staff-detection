from ultralytics import YOLO


DATA_YAML = "outputs/topdown_person_dataset_v1/data.yaml"


def main():
    model = YOLO("yolo26s.pt")

    model.train(
        data=DATA_YAML,
        epochs=30,
        imgsz=768,
        batch=8,
        device=0,
        workers=4,
        patience=8,
        project="runs/person",
        name="yolo26s_topdown",
    )


if __name__ == "__main__":
    main()