import os

# Root directory for UniversalFakeDetect (Wang2020) style layout
# Expected structure per subset: test/<subset>/{0_real,1_fake}/...
UFD_ROOT = os.environ.get('UFD_DATA_ROOT', '/path/to/CNNDetection/dataset')

def tp(*parts):
    return os.path.join(UFD_ROOT, *parts)

# Some diffusion subsets use a different real set; keep explicit mapping
_MAPPINGS = [
    ('progan',        'progan',        'progan'),
    ('cyclegan',      'cyclegan',      'cyclegan'),
    ('biggan',        'biggan',        'biggan'),
    ('stylegan',      'stylegan',      'stylegan'),
    ('gaugan',        'gaugan',        'gaugan'),
    ('stargan',       'stargan',       'stargan'),
    ('deepfake',      'deepfake',      'deepfake'),
    ('sitd',          'seeingdark',    'seeingdark'),
    ('san',           'san',           'san'),
    ('crn',           'crn',           'crn'),
    ('imle',          'imle',          'imle'),
    # diffusion / guided variants (real from imagenet or laion)
    ('guided',        'imagenet',      'guided'),
    ('ldm_200',       'laion',         'ldm_200'),
    ('ldm_200_cfg',   'laion',         'ldm_200_cfg'),
    ('ldm_100',       'laion',         'ldm_100'),
    ('glide_100_27',  'laion',         'glide_100_27'),
    ('glide_50_27',   'laion',         'glide_50_27'),
    ('glide_100_10',  'laion',         'glide_100_10'),
    ('dalle',         'laion',         'dalle'),
]

DATASET_PATHS = [
    dict(
        real_path=tp('test', real),
        fake_path=tp('test', fake),
        data_mode='wang2020',
        key=key,
    )
    for (key, real, fake) in _MAPPINGS
]
