"""Pixel transform from sukhmani1303/tuberculosis-vit-model handler.py.

Reviewed revision: 4388ed11fbfd0b0bc5f554920a9c9708485bd1d2, Apache-2.0.
The author expects OpenCV BGR; incoming PIL pixels are RGB. Keep that conversion
explicit. This function is imported only for the optional research candidate.
"""
import numpy as np


def preprocess_reference(image):
    import cv2
    rgb = np.asarray(image.convert('RGB'))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    tensor = np.moveaxis(rgb, -1, 0).astype(np.float32)
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-8)
    return np.ascontiguousarray(tensor[None])
