from natsort import natsorted
from glob import glob
import os

from tqdm import tqdm
import utils
import numpy as np

import torch


input_dir = 'shadow_test_result'#'test-results-ntire25/NTIRE25-train/DualRetiUHDM_His_illu/net_g_9600'
targrt_dir = 'tune-9600'


input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

target_paths = natsorted(
    glob(os.path.join(targrt_dir , '*.png')) + glob(os.path.join(targrt_dir , '*.jpg')))


for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(input_paths)):

    img = np.float32(utils.load_img(inp_path)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)

    tt= np.float32(utils.load_img(tar_path)) / 255.
    tt= torch.from_numpy(tt).permute(2, 0, 1)

    ss = img- tt

    print(torch.sum(ss))