# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from os import path as osp
from basicsr.utils import tensor2img, imwrite, FileClient, imfrombytes, img2tensor, padding
from basicsr.models import build_model
import time
from basicsr.utils.options import parse_options

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main(root_path):
    # parse options, set distributed setting, set ramdom seed
    opt, _ = parse_options(root_path=root_path, is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')

    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    try:
        img = imfrombytes(img_bytes, float32=True)
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)

    ## 2. set up model
    ## 2. run inference
    opt['dist'] = False
    model = build_model(opt)

    beg = time.time()
    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()

    model.test()

    if model.opt['val'].get('grids', False):
        model.grids_inverse()

    visuals = model.get_current_visuals()
    sr_img = tensor2img([visuals['result']])
    end = time.time()
    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished, \nelapsed {(end - beg):.3f} seconds. saved to {output_path}')

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    main(root_path)

