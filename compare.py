from ultralytics import YOLO
import time
from pathlib import Path


def validate_model(model_path, data_yaml, model_name):
    """Validate a model and return metrics + timing"""
    print(f"\n{'=' * 60}")
    print(f"Validating: {model_name}")
    print(f"{'=' * 60}")

    try:
        # Load model
        load_start = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start

        # Validate
        val_start = time.time()
        results = model.val(data=data_yaml, verbose=False)
        val_time = time.time() - val_start

        # Extract metrics
        metrics = {
            'name': model_name,
            'path': model_path,
            'map50_95': results.box.map,  # mAP@0.5:0.95
            'map50': results.box.map50,  # mAP@0.5
            'map75': results.box.map75,  # mAP@0.75
            'precision': results.box.mp,  # Mean precision
            'recall': results.box.mr,  # Mean recall
            'load_time': load_time,
            'val_time': val_time,
            'inference_speed': results.speed['inference'],  # ms per image
        }

        print(f"✓ Validation complete")
        return metrics

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def print_comparison(results):
    """Print formatted comparison table"""
    print(f"\n{'=' * 80}")
    print(f"MODEL COMPARISON RESULTS")
    print(f"{'=' * 80}\n")

    # Header
    print(f"{'Model':<20} {'mAP50-95':<12} {'mAP50':<10} {'Precision':<12} {'Recall':<10} {'Speed (ms)':<12}")
    print(f"{'-' * 80}")

    # Data rows
    for r in results:
        if r:
            print(f"{r['name']:<20} {r['map50_95']:<12.4f} {r['map50']:<10.4f} "
                  f"{r['precision']:<12.4f} {r['recall']:<10.4f} {r['inference_speed']:<12.2f}")

    print(f"\n{'=' * 80}")

    # Find best model for each metric
    valid_results = [r for r in results if r]
    if not valid_results:
        return

    best_map = max(valid_results, key=lambda x: x['map50_95'])
    best_speed = min(valid_results, key=lambda x: x['inference_speed'])
    best_precision = max(valid_results, key=lambda x: x['precision'])
    best_recall = max(valid_results, key=lambda x: x['recall'])

    print(f"\n🏆 WINNERS:")
    print(f"  Best Accuracy (mAP@0.5:0.95): {best_map['name']} ({best_map['map50_95']:.4f})")
    print(f"  Best Speed: {best_speed['name']} ({best_speed['inference_speed']:.2f} ms)")
    print(f"  Best Precision: {best_precision['name']} ({best_precision['precision']:.4f})")
    print(f"  Best Recall: {best_recall['name']} ({best_recall['recall']:.4f})")

    # Overall recommendation
    print(f"\n💡 RECOMMENDATION:")
    if best_map['name'] == best_speed['name']:
        print(f"  ✅ Use {best_map['name']} - Best accuracy AND speed!")
    else:
        accuracy_diff = best_map['map50_95'] - best_speed['map50_95']
        speed_diff = best_speed['inference_speed'] / best_map['inference_speed']

        print(f"  For maximum accuracy: {best_map['name']}")
        print(f"  For maximum speed: {best_speed['name']}")
        print(f"    (Speed gain: {speed_diff:.2f}x faster, Accuracy loss: {accuracy_diff:.4f})")

    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Configuration
    models_to_test = [
        ("models/tuned_model.pt", "PyTorch tuned"),
        ("models/model_fp32.onnx", "ONNX FP32"),
        ("models/updated_model.pt", "PyTorch updated"),
    ]

    data_yaml = "updated-dataset.yolo26/data.yaml"

    # Validate all models
    all_results = []
    for model_path, model_name in models_to_test:
        if Path(model_path).exists():
            result = validate_model(model_path, data_yaml, model_name)
            all_results.append(result)
        else:
            print(f"\n⚠ Warning: {model_path} not found, skipping...")
            all_results.append(None)

    # Print comparison
    print_comparison(all_results)

    # Detailed breakdown (optional)
    print(f"\nDETAILED METRICS:")
    print(f"{'-' * 80}")
    for r in all_results:
        if r:
            print(f"\n{r['name']}:")
            print(f"  mAP@0.5:0.95: {r['map50_95']:.4f}")
            print(f"  mAP@0.5:     {r['map50']:.4f}")
            print(f"  mAP@0.75:    {r['map75']:.4f}")
            print(f"  Precision:   {r['precision']:.4f}")
            print(f"  Recall:      {r['recall']:.4f}")
            print(f"  Load time:   {r['load_time']:.3f}s")
            print(f"  Val time:    {r['val_time']:.3f}s")
            print(f"  Speed:       {r['inference_speed']:.2f} ms/image")