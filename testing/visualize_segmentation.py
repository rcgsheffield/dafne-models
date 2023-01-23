#!/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import dafne.config as config
import numpy as np
import pydicom
from dafne_dl import RemoteModelProvider
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image to segment")
    parser.add_argument("model_id", help="Model ID")
    #parser.add_argument("--slice", type=int, default=0, required=False, help="Slice to show")
    parser.add_argument("--classification", nargs=1, metavar='class', type=str, required=True, help="Classification label")
    args = parser.parse_args()

    config.load_config()
    model_provider = RemoteModelProvider(config.GlobalConfig['MODEL_PATH'],
                                         config.GlobalConfig['SERVER_URL'],
                                         config.GlobalConfig['API_KEY'],
                                         config.GlobalConfig['TEMP_UPLOAD_DIR'], delete_old_models=False)

    model_classification = args.classification[0].split(',')[0]
    model = model_provider.load_model(model_classification, timestamp=args.model_id)

    # load image
    image_dataset = pydicom.dcmread(args.image_path)
    resolution = image_dataset.PixelSpacing
    image = image_dataset.pixel_array.astype(np.float32)
    input_dict = {'image': image, 'resolution': resolution, 'split_laterality': False, 'classification': model_classification}
    output = model.apply(input_dict)
    accumulated_mask = np.zeros_like(image, dtype=np.int32)
    for val, mask in enumerate(output.values()):
        accumulated_mask += mask * (val + 1)

    alpha = np.zeros_like(image, dtype=np.float32)
    alpha[accumulated_mask > 0] = 0.5

    plt.imshow(image, cmap='gray')
    plt.imshow(accumulated_mask, alpha=alpha, cmap='jet')
    plt.gca().set_axis_off()
    plt.show()
