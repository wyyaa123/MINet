# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import contextlib
import sys
import torch
import argparse
import os
import cv2 as cv
import numpy as np
from basicsr.utils import tensor2img, imwrite
from basicsr.archs.reformer_v4_arch import Reformer_v4
from basicsr.archs.reformer_v5_arch import Reformer_v5
from basicsr.archs.reformer_v1_arch import Reformer
from basicsr.archs.basefpn_arch import BaseFPNNet
from basicsr.archs.NAFNet_arch import NAFNetLocal
import time
from tqdm import tqdm

class DummyFile:
    def __init__(self, file):
        if file is None:
            file = sys.stderr
        self.file = file

    def write(self, x):
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

@contextlib.contextmanager
def redirect_stdout(file=None):
    if file is None:
        file = sys.stderr
    old_stdout = file
    sys.stdout = DummyFile(file)
    yield
    sys.stdout = old_stdout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', "-m", type=str, default='experiments/pretrained_models/BasicVSR_REDS4.pth')
    parser.add_argument('--input_path', "-i", type=str, default='datasets/REDS4/sharp_bicubic/000', help='input test image folder')
    parser.add_argument('--output_path', "-o", type=str, default='results/BasicVSR', help='save image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs_file_path = args.input_path
    output_path = args.output_path

    model = Reformer_v5(inp_ch=3,
                        width=16, 
                        patch_h=2,
                        patch_w=2,
                        middle_blk_num=8,
                        middle_use_attn=True,
                        enc_blk_nums=[2, 2, 2, 3],
                        enc_use_attns=[0, 0, 0, True],
                        dec_blk_nums=[3, 2, 2, 2],
                        dec_use_attns=[True, 0, 0, 0],
                        dw_expand=2,
                        ffn_expand=2,
                        bias=False)

    model.load_state_dict(torch.load(args.model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)

    ## 2. read images dir
    images_name = sorted(os.listdir(imgs_file_path), key=lambda name: int(''.join(filter(str.isdigit, name))))

    print ("total %d images" % len(images_name))
    time.sleep(2)
    print ("Working!.........")

    pbar = tqdm(images_name, total=len(images_name), unit="image")

    for img_name in pbar:
        pbar.set_description(f'infer {img_name}')
        img = cv.imread(imgs_file_path + img_name, cv.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        #img = img.unsqueeze(0)
        img = img.unsqueeze(0).to(device)

        ##3. inference
        beg = time.time()

        with torch.no_grad():
            output = model(img)

        sr_img = tensor2img(output)
        end = time.time()

        # imwrite(sr_img, output_path + img_name)
        cv.imwrite(sr_img, output_path + img_name[:-4] + "_deblur.png")
        with redirect_stdout():
            print(f'inference {img_name} .. finished, elapsed {(end - beg):.3f} seconds. saved to {output_path}')

    print("Done!")

if __name__ == '__main__':
    main()

