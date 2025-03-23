from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse


parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='data_shadow/test/input',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='test-results-ntire25', type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/DualRetiUHDM_His_illu.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='weights/net_g_9600.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='NTIRE25-train', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')

args = parser.parse_args()


gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################

model_restoration = create_model(opt).net_g

total = sum([param.nelement() for param in model_restoration.parameters()])
print('total parameters:', total)


checkpoint = torch.load(weights)

try:
    model_restoration.load_state_dict(checkpoint['params'])
except:
    new_checkpoint = {}
    for k in checkpoint['params']:
        new_checkpoint['module.' + k] = checkpoint['params'][k]
    model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()


factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = os.path.join(args.result_dir, dataset, config, checkpoint_name)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')
output_dir = args.output_dir
# stx()
os.makedirs(result_dir, exist_ok=True)
if args.output_dir != '':
    os.makedirs(output_dir, exist_ok=True)


input_dir = args.input_dir
print(input_dir)

input_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

target_paths = natsorted(
    glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))


with torch.inference_mode():
    for inp_path, tar_path in tqdm(zip(input_paths, target_paths), total=len(input_paths)):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path)) / 255.

        img = torch.from_numpy(img).permute(2, 0, 1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        b, c, h, w = input_.shape
        H, W = ((h + factor) // factor) * \
            factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')


        restored_R, _, _, restored_L, _, _ = model_restoration(input_)
        restored  = restored_R * restored_L
            

        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            
        if output_dir != '':
            utils.save_img((os.path.join(output_dir, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
        else:
            utils.save_img((os.path.join(result_dir, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))