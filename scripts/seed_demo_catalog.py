"""Seed two labeled scans per class and three synthetic quality tests per modality.
Selection is deterministic, never based on a model's prediction. Dataset split
labels describe the local split, not independently verified external training history.
"""
import io
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageFilter
from scripts.make_demo_assets import test_split_files
from src.database.demo_service import DemoService
from src.database.db import Database
import config


def seed(db=None):
    demos = DemoService(db or Database())
    for module, title in [('tb', 'TB'), ('dr', 'DR'), ('oral', 'Oral cancer')]:
        for label, expected in [(1, 'positive'), (0, 'negative')]:
            for i, path in enumerate(test_split_files(module, label)[:2], 1):
                demos.create(f'Demo patient {title} {expected} {i}',
                             'Fictional patient; image label from bundled dataset. ' +
                             ('Diabetes history: ' + ('type 2 (fictional).' if i == 1 else 'none (fictional).') if module == 'tb' else
                              'Diabetes status cannot be inferred from a DR label.' if module == 'dr' else 'Oral tissue histopathology.'),
                             module, expected, path.name, path.read_bytes(), age=40 + i * 8,
                             source=f'Bundled local test split: {path.relative_to(config.ROOT)}; class={path.parent.name}',
                             seed_key=f'{module}-{expected}-{i}')
        base = Image.open(test_split_files(module)[0]).convert('RGB').resize((384,384))
        for kind, im in [('blurred', base.filter(ImageFilter.GaussianBlur(35))),
                         ('blank', Image.new('RGB', (384,384), (128,128,128))),
                         ('underexposed', Image.new('RGB', (384,384), (1,1,1)))]:
            buf=io.BytesIO(); im.save(buf,format='PNG')
            demos.create(f'Demo {title} non-informative — {kind}', 'Synthetic quality rejection test; no disease label.',
                         module, 'non-informative', f'{module}_{kind}.png', buf.getvalue(),
                         source='Synthetic quality test', seed_key=f'{module}-{kind}')
    return demos

if __name__ == '__main__':
    print(f'{len(seed().list())} saved demo cases')
