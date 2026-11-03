"""Strict CPU export of the reviewed RETFound and oral research checkpoints.

Source files must be the pinned author files recorded below. Model loading is
weights-only and memory mapped; missing/unexpected parameters are fatal. No
random classifier replacement or permissive loading is allowed.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('disease', choices=['dr', 'oral'])
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--head', type=Path)
    p.add_argument('--images', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if hashlib.sha256(args.source.read_bytes()).hexdigest() != args.source_sha256:
        p.error('Reference source hash mismatch')
    if args.out.exists():
        p.error('Choose a new output directory')
    import torch
    import timm
    from PIL import Image
    from torchvision import transforms as T
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('reviewed_reference', args.source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)

    class Wrapper(torch.nn.Module):
        def __init__(self, backbone, head=None):
            super().__init__()
            self.backbone, self.head = backbone, head

        def forward(self, x):
            if self.head is None:
                # Original timm 0.3.2 forward: features then classifier. Avoid
                # newer timm's additional pooling/head normalization path.
                return self.backbone.head(self.backbone.forward_features(x))
            features, _ = self.backbone.extract_features(x)
            return self.head(features.unsqueeze(-1))

    if args.disease == 'dr':
        with torch.device('meta'):
            backbone = reference.vit_large_patch16(num_classes=5, global_pool=True)
        # The official training checkpoint stores plain argparse metadata too.
        # Allow only this known stdlib container, never arbitrary pickle code.
        with torch.serialization.safe_globals([argparse.Namespace]):
            state = torch.load(args.checkpoint, weights_only=True, mmap=True, map_location='cpu')['model']
        backbone.load_state_dict(state, strict=True, assign=True)
        model = Wrapper(backbone).eval()
        transform = T.Compose([T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                               T.CenterCrop(224), T.ToTensor(),
                               T.Normalize([.485, .456, .406], [.229, .224, .225])])
        source = 'https://github.com/rmaphoh/RETFound'
        revision = 'ad280cbf1f323d18809d3979808c4da1ae2ca040'
        labels = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']
        name, positive = 'retfound-aptos2019', 'Any diabetic retinopathy (grades 1–4)'
        license_name = 'CC-BY-NC-4.0; see retained upstream LICENSE'
    else:
        if not args.head:
            p.error('--head is required for oral')
        # Upstream constructor unnecessarily downloads ImageNet weights before
        # replacing every parameter. Disable that download; strict loading below
        # requires the complete published extractor and classifier.
        create = timm.create_model
        def create_without_download(*a, **kw):
            kw['pretrained'] = False
            return create(*a, **kw)
        timm.create_model = create_without_download
        try:
            # Swin has nonpersistent relative-position buffers; construct these
            # on CPU rather than leaving unmaterialized meta tensors.
            backbone = reference.ModifiedSwinTransformer(output_dim=1024)
            head = reference.VisionTransformer(num_classes=2, input_dim=1024)
        finally:
            timm.create_model = create
        backbone.load_state_dict(torch.load(args.checkpoint, weights_only=True, mmap=True,
                                           map_location='cpu'), strict=True, assign=True)
        head.load_state_dict(torch.load(args.head, weights_only=True, mmap=True,
                                       map_location='cpu'), strict=True, assign=True)
        model = Wrapper(backbone, head).eval()
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                               T.Normalize([.485, .456, .406], [.229, .224, .225])])
        source = 'https://huggingface.co/rudradcruze/oral-cancer'
        revision = '544e7daf03724e7725efe4fb543d749066fa2fe0'
        labels, name, positive = ['Normal', 'OSCC'], 'swin-attention-cnn-author-demo', 'OSCC'
        license_name = 'MIT; research/education only per upstream LICENSE'
    print('Strict checkpoint loading passed', flush=True)
    refs, tensors = [], []
    with torch.inference_mode():
        for path in args.images:
            with Image.open(path) as image:
                tensor = transform(image.convert('RGB')).unsqueeze(0)
            logits = model(tensor)
            if tuple(logits.shape) != (1, len(labels)) or not torch.isfinite(logits).all():
                raise ValueError('Invalid reference outputs')
            refs.append({'file': str(path.resolve()), 'raw_output': logits.flatten().tolist()})
            tensors.append(tensor)
        # Independent eager reference values above are checked against the
        # serialized model and the app's separate preprocessing implementation.
        traced = torch.jit.trace(model, tensors[0], check_trace=False, strict=True)
        for tensor, fixture in zip(tensors, refs):
            torch.testing.assert_close(traced(tensor).flatten(), torch.tensor(fixture['raw_output']),
                                       rtol=0, atol=1e-4)
    args.out.mkdir(parents=True)
    traced.save(str(args.out / 'model.pt'))
    settings = {
        'disease': args.disease, 'runtime': 'torchscript', 'model_version': name + '-' + revision[:12],
        'source': source, 'reference_revision': revision, 'license': license_name,
        'reference_source_sha256': args.source_sha256,
        'target_definition': positive, 'positive_class': positive, 'class_labels': labels,
        'positive_indices': list(range(1, len(labels))), 'output_activation': 'softmax',
        'preprocessing': {'color_mode': 'rgb', 'resize': 256 if args.disease == 'dr' else 224,
                          'resize_mode': 'shorter_side' if args.disease == 'dr' else 'exact',
                          'center_crop': 224 if args.disease == 'dr' else None, 'input_size': 224,
                          'interpolation': 'bicubic' if args.disease == 'dr' else 'bilinear',
                          'normalize': {'mean': [.485, .456, .406], 'std': [.229, .224, .225]}},
        'decision_policy': {'threshold': .5, 'uncertainty_margin': 0,
                            'validation_source': 'Research binary probability cutoff; not validated on this project population.'},
        'reference_atol': 1e-4,
    }
    (args.out / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    (args.out / 'references.json').write_text(json.dumps(refs, indent=2) + '\n')
    print('Export parity passed:', args.out, flush=True)


if __name__ == '__main__':
    main()
