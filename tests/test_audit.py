from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_data import audit, file_key, find_image_mask_directories, pair_image_masks


def test_matching_and_shape_audit(tmp_path):
    images = tmp_path / 'nested' / 'BUSBRA' / 'Images'
    masks = images.parent / 'Masks'
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    Image.fromarray(np.full((12, 20), 90, dtype=np.uint8)).save(images / 'bus_0003.png')
    x = np.zeros((12, 20), dtype=np.uint8)
    x[2:6, 3:9] = 255
    Image.fromarray(x).save(masks / 'mask_0003.png')
    (images.parent / '5-fold-cv.csv').write_text('ID,kFold\nbus_0003,1\n')
    result = audit(tmp_path)
    assert result['paired_count'] == 1
    assert len(result['shape_mismatches']) == 0
    assert len(result['csv_metadata']) == 1
    assert result['image_dimensions']['20x12'] == 1


def test_keys_preserve_pairing():
    assert file_key(Path('bus_000134.png')) == file_key(Path('mask_000134.png'))
