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

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        r = model(x)[0]
        l = model(x)[3]
        if rotate:
            r = torch.rot90(r, dims=(-2, -1), k=3)
            l = torch.rot90(l, dims=(-2, -1), k=3)
        if vflip:
            r = torch.flip(r, (-1,))
            l = torch.flip(l, (-1,))
        if hflip:
            r = torch.flip(r, (-2,))
            l = torch.flip(l, (-2,))
        return r, l
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                rr, ll = forward_transformed(x, hflip, vflip, rot, model)
                t.append(rr*ll)
                t
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='data_25/test/input',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='25test-results-ensemble', type=str, help='Directory for results')
parser.add_argument('--output_dir', default='',
                    type=str, help='Directory for output')
parser.add_argument(
    '--opt', type=str, default='Options/DualRetiUHDM_His_illu_24train_2loss_mulitistep_2x900-finetune.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='experiments/DualRetiUHDM_His_illu_24train_2loss_mulitistep_2x900-finetune/models/net_g_15000.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='24train', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')

args = parser.parse_args()

# 指定 gpu
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

# 加载模型
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

# 生成输出结果的文件
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

        if h < 3000 and w < 3000:
            restored = self_ensemble(input_, model_restoration)
        else:
            input_1 = input_[:, :, :, 1::2]
            input_2 = input_[:, :, :, 0::2]
            
            restored_1 = self_ensemble(input_1, model_restoration)
            restored_2 = self_ensemble(input_2, model_restoration)
                  
            restored = torch.zeros_like(input_)
            restored[:, :, :, 1::2] = restored_1
            restored[:, :, :, 0::2] = restored_2
            

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
