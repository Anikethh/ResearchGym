import yaml

# 原始YAML文件路径
original_yaml_path = 'retrieval_flickr.yaml'

replacements = [
    'Flickr30K_IP_brightness_1',
    'Flickr30K_IP_brightness_2',
    'Flickr30K_IP_brightness_3',
    'Flickr30K_IP_brightness_4',
    'Flickr30K_IP_brightness_5',
    'Flickr30K_IP_contrast_1',
    'Flickr30K_IP_contrast_2',
    'Flickr30K_IP_contrast_3',
    'Flickr30K_IP_contrast_4',
    'Flickr30K_IP_contrast_5',
    'Flickr30K_IP_defocus_blur_1',
    'Flickr30K_IP_defocus_blur_2',
    'Flickr30K_IP_defocus_blur_3',
    'Flickr30K_IP_defocus_blur_4',
    'Flickr30K_IP_defocus_blur_5',
    'Flickr30K_IP_elastic_transform_1',
    'Flickr30K_IP_elastic_transform_2',
    'Flickr30K_IP_elastic_transform_3',
    'Flickr30K_IP_elastic_transform_4',
    'Flickr30K_IP_elastic_transform_5',
    'Flickr30K_IP_fog_1',
    'Flickr30K_IP_fog_2',
    'Flickr30K_IP_fog_3',
    'Flickr30K_IP_fog_4',
    'Flickr30K_IP_fog_5',
    'Flickr30K_IP_frost_1',
    'Flickr30K_IP_frost_2',
    'Flickr30K_IP_frost_3',
    'Flickr30K_IP_frost_4',
    'Flickr30K_IP_frost_5',
    'Flickr30K_IP_gaussian_noise_1',
    'Flickr30K_IP_gaussian_noise_2',
    'Flickr30K_IP_gaussian_noise_3',
    'Flickr30K_IP_gaussian_noise_4',
    'Flickr30K_IP_gaussian_noise_5',
    'Flickr30K_IP_glass_blur_1',
    'Flickr30K_IP_glass_blur_2',
    'Flickr30K_IP_glass_blur_3',
    'Flickr30K_IP_glass_blur_4',
    'Flickr30K_IP_glass_blur_5',
    'Flickr30K_IP_impulse_noise_1',
    'Flickr30K_IP_impulse_noise_2',
    'Flickr30K_IP_impulse_noise_3',
    'Flickr30K_IP_impulse_noise_4',
    'Flickr30K_IP_impulse_noise_5',
    'Flickr30K_IP_jpeg_compression_1',
    'Flickr30K_IP_jpeg_compression_2',
    'Flickr30K_IP_jpeg_compression_3',
    'Flickr30K_IP_jpeg_compression_4',
    'Flickr30K_IP_jpeg_compression_5',
    'Flickr30K_IP_motion_blur_1',
    'Flickr30K_IP_motion_blur_2',
    'Flickr30K_IP_motion_blur_3',
    'Flickr30K_IP_motion_blur_4',
    'Flickr30K_IP_motion_blur_5',
    'Flickr30K_IP_pixelate_1',
    'Flickr30K_IP_pixelate_2',
    'Flickr30K_IP_pixelate_3',
    'Flickr30K_IP_pixelate_4',
    'Flickr30K_IP_pixelate_5',
    'Flickr30K_IP_shot_noise_1',
    'Flickr30K_IP_shot_noise_2',
    'Flickr30K_IP_shot_noise_3',
    'Flickr30K_IP_shot_noise_4',
    'Flickr30K_IP_shot_noise_5',
    'Flickr30K_IP_snow_1',
    'Flickr30K_IP_snow_2',
    'Flickr30K_IP_snow_3',
    'Flickr30K_IP_snow_4',
    'Flickr30K_IP_snow_5',
    'Flickr30K_IP_speckle_noise_1',
    'Flickr30K_IP_speckle_noise_2',
    'Flickr30K_IP_speckle_noise_3',
    'Flickr30K_IP_speckle_noise_4',
    'Flickr30K_IP_speckle_noise_5',
    'Flickr30K_IP_zoom_blur_1',
    'Flickr30K_IP_zoom_blur_2',
    'Flickr30K_IP_zoom_blur_3',
    'Flickr30K_IP_zoom_blur_4',
    'Flickr30K_IP_zoom_blur_5'
]

keyword = 'flickr30k-images-test'

# 读取原始YAML文件
with open(original_yaml_path, 'r') as file:
    yaml_content = file.read()  # 直接读取原始内容，因为yaml.safe_load可能无法处理所有YAML结构

# 遍历替换列表，生成新的YAML文件
for i, replacement in enumerate(replacements):
    # 替换关键字
    # replacement=f'flickr30k-IP/{replacement}'
    new_content = yaml_content.replace(keyword, f'flickr30k-IP/{replacement}')
    
    # 构建新的YAML文件名
    new_yaml_name = f"{replacement.replace(' ', '_')}.yaml"
    
    # 写入新的YAML文件
    with open(new_yaml_name, 'w') as new_file:
        new_file.write(new_content)

    print(f"Generated {new_yaml_name}")