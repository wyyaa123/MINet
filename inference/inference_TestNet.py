# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import argparse
import os
import cv2 as cv
import numpy as np
from basicsr.utils import tensor2img, imwrite
from basicsr.archs.Testline_v2_arch import TestFPNNet_v2
import time

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main():
    # parse options, set distributed setting, set ramdom seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/BasicVSR_REDS4.pth')
    parser.add_argument('--input_path', type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--output_path', type=str, default='results/BasicVSR', help='save image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = args.input_path
    output_path = args.output_path


    ## 1. read image
    img = cv.imread(img_path, cv.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # img = img.unsqueeze(0)
    img = img.unsqueeze(0).to(device)

    ## 2. set up model
    model = TestFPNNet_v2(3)
    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    ##3. inference
    beg = time.time()

    with torch.no_grad():
        output = model(img)
    
    sr_img = tensor2img(output)
    end = time.time()

    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished, \nelapsed {(end - beg):.3f} seconds. saved to {output_path}')

if __name__ == '__main__':
    main()

