"""Package the pinned public DenseNet TB candidate using its author's inference.

Download model.onnx separately from the revision below. Two existing research
images are used only for software parity, not clinical validation or tuning.
"""
import argparse
import json
from pathlib import Path

REVISION = '6d9b27d040e51ef2bad4906e8fb97132b09d265d'
SOURCE = 'https://huggingface.co/Ressshh/tb-conformal-distilled-densenet121'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.model), sess_options=opts, providers=['CPUExecutionProvider'])
    references = []
    # Author quickstart: PIL's default RGB resize is bicubic. No grayscale,
    # border crop, routing confidence, or quality multiplier is applied.
    for path in args.images:
        with Image.open(path) as original:
            image = original.convert('RGB').resize((224, 224))
        arr = np.array(image).astype(np.float32) / 255.0
        arr = ((arr - np.array([.485, .456, .406], dtype=np.float32)) /
               np.array([.229, .224, .225], dtype=np.float32)).transpose(2, 0, 1)[None]
        logits = session.run(None, {session.get_inputs()[0].name: arr})[0]
        references.append({'file': str(path.resolve()), 'raw_output': logits.reshape(-1).tolist()})
    settings = {
        'disease': 'tb', 'runtime': 'onnx',
        'model_version': 'biomedclip-distilled-densenet121-v11-' + REVISION[:12],
        'source': SOURCE, 'reference_revision': REVISION, 'license': 'MIT (author model card)',
        'target_definition': 'Pulmonary tuberculosis on chest radiographs',
        'positive_class': 'Tuberculosis', 'class_labels': ['Normal', 'Tuberculosis'],
        'positive_indices': [1], 'output_activation': 'softmax',
        'preprocessing': {'color_mode': 'rgb', 'resize': 224, 'resize_mode': 'exact',
                          'center_crop': None, 'input_size': 224, 'interpolation': 'bicubic',
                          'normalize': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}},
        'decision_policy': {'threshold': .5, 'uncertainty_margin': 0,
                            'validation_source': 'Research argmax policy; not validated on this project population. Author cohort thresholds are not transferred.'},
        'reference_atol': 1e-5,
    }
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (args.out / 'references.json').write_text(json.dumps(references, indent=2) + '\n')
    print(args.out)


if __name__ == '__main__':
    main()
