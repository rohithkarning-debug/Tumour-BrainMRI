import sys, torch, numpy as np
sys.path.insert(0, '.')
from src.preprocess import TrainingPreprocessor

prep = TrainingPreprocessor(spatial_size=(64,64,64))
dummy_image = np.random.randn(150, 180, 150).astype(np.float32)
dummy_mask  = np.zeros((150, 180, 150), dtype=np.float32)
dummy_mask[70:80, 85:95, 70:80] = 1.0

sample = {'image': dummy_image, 'mask': dummy_mask, 'patient_id': 'test',
          'image_path': '', 'mask_path': '', 'has_tumour': True}

for i in range(5):
    out = prep(sample)
    assert out['image'].shape == (1,64,64,64), f'Bad shape: {out["image"].shape}'
    assert out['mask'].shape  == (1,64,64,64)
    assert isinstance(out['patient_id'], str)
    assert isinstance(out['mask_path'], str)

print('ALL 5 AUGMENTATION PASSES OK')
print(f'image shape: {out["image"].shape}')
print(f'mask shape:  {out["mask"].shape}')
print(f'has_tumour:  {out["has_tumour"]}')
print(f'mask_path:   "{out["mask_path"]}"')
