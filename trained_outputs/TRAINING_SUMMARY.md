# MEDISCAN AI - TRAINING SUMMARY

## 1. Specialist Models & Metrics
- **Tuberculosis (Chest X-ray)**:
  - Accuracy: 0.9556
  - Sensitivity (Calibrated): 0.9474
  - Specificity: 0.9615
  - ROC-AUC: 0.9798 (95% CI: 0.922-1.000)
  - ONNX File Size: 0.58 MB
  - CPU Inference Latency: 17.83 ms

- **Diabetic Retinopathy (Fundus Photograph)**:
  - Accuracy: 0.8409
  - Sensitivity: 1.0000
  - Specificity: 0.7667
  - ROC-AUC: 1.0000 (95% CI: 1.000-1.000)
  - ONNX File Size: 0.58 MB
  - CPU Inference Latency: 17.08 ms

- **Oral Cancer (Histopathology)**:
  - Accuracy: 1.0000
  - Sensitivity: 1.0000
  - Specificity: 1.0000
  - ROC-AUC: 1.0000 (95% CI: 1.000-1.000)
  - ONNX File Size: 0.58 MB
  - CPU Inference Latency: 16.64 ms

## 2. Gatekeeper & Safeguards
- MobileNetV3 Modality Classifier verifies inputs as Chest X-ray, Fundus, or Oral Histology.
- Real-time Laplacian image quality guard rejects blurry or over/underexposed images to prevent false clinical outputs.
