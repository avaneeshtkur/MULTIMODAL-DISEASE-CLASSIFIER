"""One immutable model bundle per worker, verified against reference fixtures."""
import hashlib
import importlib.util
import json
import math
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from src.core.image_preprocessor import preprocess, validate_spec
from src.core.inference_engine import decide, uncertainty_band
from src.providers.remote_research import number
from src.utils.image_hash import sha256_file


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class ResearchModel:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.spec = json.loads((self.folder / 'manifest.json').read_text())
        s = self.spec
        if s.get('schema_version') != '1.0' or s.get('disease') not in ('tb', 'dr', 'oral'):
            raise ValueError('Invalid research manifest')
        for key in ('model_version', 'target_definition', 'positive_class', 'source', 'license', 'reference_revision'):
            if not isinstance(s.get(key), str) or not s[key].strip():
                raise ValueError(f'Missing model provenance: {key}')
        if s.get('runtime') not in ('onnx', 'torchscript', 'tbnet_tf'):
            raise ValueError('Unsupported runtime')
        if not isinstance(s.get('files'), dict) or not s['files']:
            raise ValueError('Model files and hashes are required')
        for name, expected in s['files'].items():
            path = self.path(name)
            if sha256_file(path) != expected:
                raise ValueError(f'Artifact hash mismatch: {name}')
        if s['model_file'] not in s['files']:
            raise ValueError('Primary model must be pinned')
        # Bundle identity binds all model shards and any executable reference preprocessing.
        self.model_hash = digest({k: s.get(k) for k in ('files', 'class_labels', 'positive_indices', 'output_activation')})
        classes = s.get('class_labels')
        positive = s.get('positive_indices')
        if not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError('Explicit unique class labels required')
        if not isinstance(positive, list) or not positive or len(set(positive)) != len(positive) or any(type(i) is not int or not 0 <= i < len(classes) for i in positive) or len(positive) == len(classes):
            raise ValueError('Positive class indices must identify a proper subset of classes')
        if s.get('output_activation') not in ('sigmoid', 'softmax', 'positive_probability'):
            raise ValueError('Declare logits activation or scalar positive probability')
        if s['output_activation'] != 'softmax' and (len(classes) != 2 or len(positive) != 1):
            raise ValueError('Scalar outputs require a binary target')
        policy = s.get('decision_policy', {})
        number(policy.get('threshold'))
        if not 0 < policy['threshold'] < 1:
            raise ValueError('Decision threshold must lie strictly between zero and one')
        if policy.get('uncertainty_margin') is not None:
            number(policy['uncertainty_margin'])
        else:
            number(policy.get('uncertainty_margin_logit'), 0, 100)
        if not isinstance(policy.get('validation_source'), str) or not policy['validation_source'].strip():
            raise ValueError('Decision policy must document its validation source')
        temperature = policy.get('temperature', 1)
        if not math.isfinite(temperature) or not 0 < temperature <= 100:
            raise ValueError('Invalid temperature')
        self.policy = policy
        if s['runtime'] == 'tbnet_tf':
            if s['disease'] != 'tb' or s.get('preprocessing') != {'kind': 'tbnet_reference'}:
                raise ValueError('TB-Net requires its exact reference preprocessing')
        elif s.get('preprocessing') == {'kind': 'tb_vit_clahe_v1'}:
            if s['disease'] != 'tb' or s['runtime'] != 'torchscript':
                raise ValueError('TB ViT reference preprocessing requires a TB TorchScript bundle')
        else:
            validate_spec(s['preprocessing'])
        self.identity = dict(model_version=s['model_version'], model_sha256=self.model_hash,
                             target_definition=s['target_definition'], positive_class=s['positive_class'],
                             preprocessing_version=digest(s['preprocessing']), decision_policy_version=digest(policy))
        self._load()
        # Fail closed unless end-to-end pixel → logits parity holds on >=2 pinned fixtures.
        fixtures = s.get('reference_fixtures', [])
        if len(fixtures) < 2 or len({f.get('sha256') for f in fixtures}) < 2:
            raise ValueError('At least two distinct independently generated reference fixtures are required')
        tolerance = number(s.get('reference_atol', 1e-5), 0, 0.001)
        for fixture in fixtures:
            path = self.path(fixture['file'])
            if sha256_file(path) != fixture['sha256']:
                raise ValueError('Reference image hash mismatch')
            with Image.open(path) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError('Reference fixture exceeds pixel limit')
                actual = self.raw(image.convert('RGB'))
            expected = np.asarray(fixture['raw_output'], dtype=float)
            if actual.shape != expected.shape or not np.isfinite(expected).all() or not np.allclose(actual, expected, rtol=0, atol=tolerance):
                raise ValueError('Reference inference does not reproduce; model remains disabled')
            self.score(actual)  # validates output width, class mapping and numeric range

    def path(self, name):
        path = (self.folder / name).resolve()
        if not path.is_relative_to(self.folder) or not path.is_file():
            raise ValueError('Bundle file is missing or outside the bundle')
        return path

    def _load(self):
        s = self.spec
        model_path = self.path(s['model_file'])
        if s['runtime'] == 'onnx':
            import onnx
            import onnxruntime as ort
            graph = onnx.load(str(model_path), load_external_data=False)
            # External weights must be explicitly pinned inside the bundle too.
            for tensor in graph.graph.initializer:
                for entry in tensor.external_data:
                    if entry.key == 'location' and entry.value not in s['files']:
                        raise ValueError('Unpinned external ONNX weights')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            self.model = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
            if len(self.model.get_inputs()) != 1 or len(self.model.get_outputs()) != 1:
                raise ValueError('Research ONNX model must have exactly one input/output')
        elif s['runtime'] == 'torchscript':
            import torch
            torch.set_num_threads(2)
            self.model = torch.jit.load(str(model_path), map_location='cpu').eval()
        else:
            import tensorflow.compat.v1 as tf
            tf.disable_eager_execution()
            source = s.get('preprocessing_file')
            if source not in s['files']:
                raise ValueError('Pin the reviewed TB-Net reference preprocessing source')
            module_spec = importlib.util.spec_from_file_location('tbnet_reference', self.path(source))
            self.reference = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(self.reference)
            self.graph = tf.Graph()
            with self.graph.as_default():
                saver = tf.train.import_meta_graph(str(model_path))
            prefix = s.get('checkpoint_prefix', 'model')
            for name in (prefix + '.index', prefix + '.data-00000-of-00001'):
                if name not in s['files']:
                    raise ValueError('Missing pinned TB-Net checkpoint shard')
            self.model = tf.Session(graph=self.graph, config=tf.ConfigProto(intra_op_parallelism_threads=2, inter_op_parallelism_threads=1))
            saver.restore(self.model, str(self.folder / prefix))

    def raw(self, image):
        runtime = self.spec['runtime']
        if runtime == 'tbnet_tf':
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / 'scan.png'
                image.save(path)
                x = self.reference.preprocess_image_inference(str(path))
            out = self.model.run(self.graph.get_tensor_by_name('resnet_model/final_dense:0'),
                                 {self.graph.get_tensor_by_name('image:0'): [x]})
        else:
            if self.spec['preprocessing'] == {'kind': 'tb_vit_clahe_v1'}:
                from services.research.tb_vit import preprocess_reference
                x = preprocess_reference(image)
            else:
                x = preprocess(image, self.spec['preprocessing'])
            if runtime == 'onnx':
                out = self.model.run(None, {self.model.get_inputs()[0].name: x})[0]
            else:
                import torch
                with torch.inference_mode():
                    out = self.model(torch.from_numpy(x)).detach().cpu().numpy()
        out = np.asarray(out, dtype=float).reshape(-1)
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite model output')
        return out

    def score(self, raw, temperature=1):
        activation = self.spec['output_activation']
        if activation == 'softmax':
            if len(raw) != len(self.spec['class_labels']):
                raise ValueError('Output width does not match declared classes')
            z = raw / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            return float(p[self.spec['positive_indices']].sum())
        if len(raw) != 1:
            raise ValueError('Expected scalar output')
        z = float(raw[0])
        if activation == 'positive_probability':
            number(z)
            if temperature == 1:
                return z
            z = float(np.log(np.clip(z, 1e-7, 1-1e-7) / (1 - np.clip(z, 1e-7, 1-1e-7))))
        return float(np.exp(-np.logaddexp(0, -z / temperature)))

    def predict(self, image):
        raw = self.raw(image)
        original = self.score(raw)
        score = self.score(raw, self.policy.get('temperature', 1))
        status, code = decide(score, self.policy)
        label = self.spec['positive_class'] if status == 'SCREENING_FLAGGED' else (
            ', '.join(c for i, c in enumerate(self.spec['class_labels']) if i not in self.spec['positive_indices'])
            if status == 'NO_FLAG_DETECTED' else None)
        return dict(model_score=score, raw_positive_score=original, raw_output=raw.tolist(),
                    threshold=self.policy['threshold'], uncertainty_band=list(uncertainty_band(self.policy)),
                    prediction_status=status, class_label=label,
                    reason_code=code, reason='Within the decision uncertainty band.' if code else '',
                    score_kind='temperature-scaled model score' if self.policy.get('temperature', 1) != 1 else 'uncalibrated model score',
                    provider_status='completed', **self.identity)
