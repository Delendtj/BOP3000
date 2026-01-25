from ultralytics import YOLO

if __name__ == "__main__":
    # Paths to models
    model_a_path = "models/updated_model.pt"
    model_b_path = "models/"

    # Dataset YAML (must be the SAME)
    data_yaml = "updated-dataset-m-f.yolo26/data.yaml"

    # Load models
    model_a = YOLO(model_a_path)
    model_b = YOLO(model_b_path)

    # Validate
    results_a = model_a.val(data=data_yaml)
    results_b = model_b.val(data=data_yaml)

    # Extract metrics
    map_a = results_a.box.map      # mAP@0.5:0.95
    map_b = results_b.box.map

    map50_a = results_a.box.map50
    map50_b = results_b.box.map50

    # Compare
    print(f"Model A mAP@0.5:0.95: {map_a:.4f}")
    print(f"Model B mAP@0.5:0.95: {map_b:.4f}")

    if map_a > map_b:
        print("✅ Model A is better")
    elif map_b > map_a:
        print("✅ Model B is better")
    else:
        print("🤝 Models are equal")
